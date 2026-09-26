//! Crash-recovery integration test (CM-07 WP-07.3 S0).
//!
//! Scenario: 3 workers in a pool. w1 is the initial coordinator and
//! posts epoch 0, then stalls (crashed or censoring).
//!
//! Current `ComputePoolTraining` rules:
//!   - Only the requester or governance may appoint a new coordinator
//!     (`reassignCoordinator`); a joined worker cannot.
//!   - A requester swap after COORDINATION_TIMEOUT is a plain
//!     replacement; the liveness slash applies only when governance
//!     adjudicates the stall.
//!   - The worker-side exit is `expireStalledTraining` after
//!     STALL_EXPIRY_BLOCKS of inactivity: the job moves to Awaiting,
//!     the challenge window runs from expiry, then it finalizes.
//!
//! Chain layer is `MockChainClient` — same invariants as the
//! Solidity ComputePoolTraining contract.

use std::sync::Arc;

use citrate_training_worker::{
    chain::{ChainError, JobChainState, MOCK_REQUESTER, STALL_EXPIRY_BLOCKS},
    ChainClient, MockChainClient, TrainingJobSpec,
};
use ethereum_types::{Address, H256};

fn make_spec() -> TrainingJobSpec {
    TrainingJobSpec {
        model_start_hash: H256::repeat_byte(0x11),
        dataset_hash: H256::repeat_byte(0x22),
        epoch_count: 2,
        steps_per_epoch: 2,
        min_workers: 3,
        max_workers: 5,
        challenge_window_blocks: 10,
        per_epoch_budget: 30_000_000_000_000_000_000u128,
        per_worker_stake: 10_000_000_000_000_000_000u128,
    }
}

async fn stalled_job(chain: &MockChainClient, spec: &TrainingJobSpec) -> (u64, [Address; 3]) {
    let job_id = chain.create_job(spec.clone());
    let ws = [
        Address::repeat_byte(1),
        Address::repeat_byte(2),
        Address::repeat_byte(3),
    ];
    for w in ws {
        chain
            .join_training_job(job_id, w, spec.per_worker_stake)
            .await
            .expect("join");
    }
    chain
        .close_recruitment(job_id, ws[0])
        .await
        .expect("close recruitment");
    // w1 (original coordinator) posts epoch 0 normally, then stalls.
    chain
        .commit_epoch(job_id, ws[0], 0, H256::repeat_byte(0xE0))
        .await
        .expect("epoch 0 commit");
    (job_id, ws)
}

#[tokio::test]
async fn requester_reassigns_stalled_coordinator_and_job_finalizes() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let spec = make_spec();
    let (job_id, [w1, _w2, w3]) = stalled_job(&chain, &spec).await;

    // Under COORDINATION_TIMEOUT: the requester's swap reverts.
    chain.advance_blocks(50).await;
    let err = chain
        .reassign_coordinator(job_id, MOCK_REQUESTER, w3)
        .await
        .expect_err("reassign must reject before timeout");
    assert!(matches!(err, ChainError::CoordinatorStillActive));

    // Past the timeout the requester appoints w3 (no slash on a
    // requester swap).
    chain.advance_blocks(51).await;
    chain
        .reassign_coordinator(job_id, MOCK_REQUESTER, w3)
        .await
        .expect("requester reassign past timeout");
    assert!(!chain.was_liveness_slashed(job_id, w1));

    let snap = chain.snapshot(job_id).await.expect("snapshot");
    assert_eq!(snap.coordinator, Some(w3));
    assert_eq!(snap.current_epoch, 1);

    chain
        .commit_epoch(job_id, w3, 1, H256::repeat_byte(0xE1))
        .await
        .expect("epoch 1 commit by new coordinator");
    chain
        .advance_blocks(spec.challenge_window_blocks as u64 + 1)
        .await;
    chain.finalize(job_id).await.expect("finalize");

    let snap = chain.snapshot(job_id).await.expect("snapshot");
    assert_eq!(snap.state, JobChainState::Finalized);
    assert_eq!(snap.epoch_roots.len(), 2);
}

#[tokio::test]
async fn joined_worker_cannot_appoint_a_coordinator() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let spec = make_spec();
    let (job_id, [w1, w2, w3]) = stalled_job(&chain, &spec).await;
    chain.advance_blocks(200).await;

    for (caller, pick) in [(w2, w3), (w2, w2)] {
        let err = chain
            .reassign_coordinator(job_id, caller, pick)
            .await
            .expect_err("a worker must not appoint the coordinator");
        assert!(matches!(err, ChainError::NotRequesterOrGovernance));
    }
    assert_eq!(
        chain.snapshot(job_id).await.expect("snapshot").coordinator,
        Some(w1)
    );
}

#[tokio::test]
async fn worker_expires_a_stalled_job_and_it_finalizes() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let spec = make_spec();
    let (job_id, [_w1, w2, _w3]) = stalled_job(&chain, &spec).await;

    chain.advance_blocks(STALL_EXPIRY_BLOCKS).await;
    let err = chain
        .expire_stalled_training(job_id, w2)
        .await
        .expect_err("not stalled until STALL_EXPIRY_BLOCKS have fully passed");
    assert!(matches!(err, ChainError::NotStalled));

    chain.advance_blocks(1).await;
    chain
        .expire_stalled_training(job_id, w2)
        .await
        .expect("joined worker expires the stalled job");
    let snap = chain.snapshot(job_id).await.expect("snapshot");
    assert_eq!(snap.state, JobChainState::Awaiting);
    assert_eq!(snap.epoch_roots.len(), 1, "committed epoch stays on record");

    // The challenge window runs from expiry before anything settles.
    assert!(matches!(
        chain.finalize(job_id).await,
        Err(ChainError::ChallengeWindowOpen)
    ));
    chain
        .advance_blocks(spec.challenge_window_blocks as u64)
        .await;
    chain.finalize(job_id).await.expect("finalize after expiry");
    assert_eq!(
        chain.snapshot(job_id).await.expect("snapshot").state,
        JobChainState::Finalized
    );
}

#[tokio::test]
async fn non_joined_caller_cannot_expire_or_reassign() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let spec = make_spec();
    let (job_id, [_w1, _w2, w3]) = stalled_job(&chain, &spec).await;
    chain.advance_blocks(STALL_EXPIRY_BLOCKS + 1).await;

    let outsider = Address::repeat_byte(0xEE);
    let err = chain
        .reassign_coordinator(job_id, outsider, w3)
        .await
        .expect_err("must reject non-joined caller");
    assert!(matches!(err, ChainError::NotRequesterOrGovernance));
    let err = chain
        .expire_stalled_training(job_id, outsider)
        .await
        .expect_err("outsider cannot expire");
    assert!(matches!(err, ChainError::NotRequesterOrGovernance));
}
