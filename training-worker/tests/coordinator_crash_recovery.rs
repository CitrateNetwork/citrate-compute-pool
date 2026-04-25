//! Crash-recovery integration test (CM-07 WP-07.3 S0).
//!
//! Scenario: 3 workers in a pool. w1 is the initial coordinator.
//! w1 posts epoch 0 root successfully. Then w1 stalls — either
//! crashed, or deliberately censoring. After COORDINATION_TIMEOUT
//! blocks with no activity, w2 calls `reassign_coordinator` to
//! elect w3. w3 posts epoch 1 root, job finalizes.
//!
//! Asserts:
//!   - Reassignment rejected before timeout elapses
//!   - Reassignment accepted after timeout
//!   - Old coordinator (w1) is liveness-slashed
//!   - New coordinator (w3) can successfully post commitEpoch
//!   - Job reaches Finalized terminal state
//!
//! Chain layer is `MockChainClient` — same invariants as the
//! Solidity ComputePoolTraining contract. The worker-side code
//! path isn't exercised end-to-end here; that's WP-07.3 S1 scope
//! (needs the Worker type refactored to support role transitions
//! mid-run). What this test DOES validate is the chain contract
//! that any S1 refactor will consume.

use std::sync::Arc;

use citrate_training_worker::{
    chain::{ChainError, JobChainState},
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

#[tokio::test]
async fn coordinator_stall_triggers_reassignment_and_job_finalizes() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let spec = make_spec();
    let job_id = chain.create_job(spec.clone());

    let w1 = Address::repeat_byte(1);
    let w2 = Address::repeat_byte(2);
    let w3 = Address::repeat_byte(3);

    for w in [w1, w2, w3] {
        chain
            .join_training_job(job_id, w, spec.per_worker_stake)
            .await
            .expect("join");
    }
    chain
        .close_recruitment(job_id, w1)
        .await
        .expect("close recruitment");

    // w1 (original coordinator) posts epoch 0 normally.
    chain
        .commit_epoch(job_id, w1, 0, H256::repeat_byte(0xE0))
        .await
        .expect("epoch 0 commit");

    // w1 stalls. Roll only 50 blocks (under COORDINATION_TIMEOUT) —
    // reassignment should revert.
    chain.advance_blocks(50).await;
    let err = chain
        .reassign_coordinator(job_id, w2, w3)
        .await
        .expect_err("reassign must reject before timeout");
    assert!(matches!(err, ChainError::CoordinatorStillActive));

    // Roll past the timeout threshold (total 101 blocks since last
    // activity). w2 calls reassign → w3.
    chain.advance_blocks(51).await;
    chain
        .reassign_coordinator(job_id, w2, w3)
        .await
        .expect("reassign past timeout");

    // w1 was liveness-slashed.
    assert!(
        chain.was_liveness_slashed(job_id, w1),
        "old coordinator must be liveness-slashed"
    );

    // w3 is the new coordinator. They can now post epoch 1.
    let snap = chain.snapshot(job_id).await.expect("snapshot");
    assert_eq!(snap.coordinator, Some(w3));
    assert_eq!(snap.current_epoch, 1);

    chain
        .commit_epoch(job_id, w3, 1, H256::repeat_byte(0xE1))
        .await
        .expect("epoch 1 commit by new coordinator");

    // Job is Awaiting — wait out challenge window, finalize.
    chain
        .advance_blocks(spec.challenge_window_blocks as u64 + 1)
        .await;
    chain.finalize(job_id).await.expect("finalize");

    let snap = chain.snapshot(job_id).await.expect("snapshot");
    assert_eq!(snap.state, JobChainState::Finalized);
    // Both epoch roots on-chain.
    assert_eq!(snap.epoch_roots.len(), 2);
}

#[tokio::test]
async fn non_joined_worker_cannot_trigger_reassignment() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let spec = make_spec();
    let job_id = chain.create_job(spec.clone());

    let w1 = Address::repeat_byte(1);
    let w2 = Address::repeat_byte(2);
    let w3 = Address::repeat_byte(3);
    for w in [w1, w2, w3] {
        chain
            .join_training_job(job_id, w, spec.per_worker_stake)
            .await
            .unwrap();
    }
    chain.close_recruitment(job_id, w1).await.unwrap();
    chain.advance_blocks(200).await;

    let outsider = Address::repeat_byte(0xEE);
    let err = chain
        .reassign_coordinator(job_id, outsider, w3)
        .await
        .expect_err("must reject non-joined caller");
    assert!(matches!(err, ChainError::CallerNotJoined));
}
