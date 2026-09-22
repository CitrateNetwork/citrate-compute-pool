//! End-to-end integration test — 3 workers × 2 epochs × 2 steps
//! against the `MockChainClient` and `InProcessTransport`.
//!
//! This is the S0 acceptance test from the CM-07/08 staged execution
//! plan. It validates:
//!
//!  - Workers independently produce deterministic step commits
//!  - Coordinator aggregates commits into a Merkle root and posts
//!    `commitEpoch`
//!  - Non-coordinator workers advance past each epoch only after
//!    the coordinator's root lands on-chain
//!  - Job transitions Recruiting → Training → Awaiting → Finalized
//!  - All three workers return `Finalized` outcome
//!
//! The mock chain client enforces the same state machine the
//! Solidity contract does (epoch monotonicity, coordinator-only
//! `commitEpoch`, challenge-window gate on `finalize`), so any
//! state-machine bug in the worker would fail this test the same
//! way it would fail against a live deployment.

use std::sync::Arc;

use citrate_training_worker::{
    chain::JobChainState, ChainClient, DeterministicTinyModel, InProcessTransport, MockChainClient,
    TrainingJobSpec, Transport, Worker, WorkerConfig, WorkerOutcome,
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
async fn three_workers_two_epochs_two_steps_end_to_end() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .try_init();

    let chain = MockChainClient::new();
    let transport = InProcessTransport::new();
    let backend = Arc::new(DeterministicTinyModel::new());

    // Create the job on-chain (mock), then open it and recruit
    // three workers.
    let spec = make_spec();
    let job_id = chain.create_job(spec.clone());

    let w1 = Address::repeat_byte(1);
    let w2 = Address::repeat_byte(2);
    let w3 = Address::repeat_byte(3);

    for worker_addr in [w1, w2, w3] {
        chain
            .join_training_job(job_id, worker_addr, spec.per_worker_stake)
            .await
            .expect("join");
        transport.register(worker_addr).await;
    }

    // Close recruitment — w1 is the coordinator for the whole job.
    // Per-epoch VRF rotation is a future slice; S0 fixes the
    // coordinator at close time.
    chain.close_recruitment(job_id, w1).await.expect("close");

    // Build three Worker structs, each with their own scoped
    // transport so broadcasts skip their own queue.
    let worker1 = Worker::new(
        WorkerConfig {
            job_id,
            self_address: w1,
            shard_index: 0,
            is_coordinator: true,
        },
        Arc::clone(&backend),
        transport.scoped(w1),
        Arc::clone(&chain),
    );
    let worker2 = Worker::new(
        WorkerConfig {
            job_id,
            self_address: w2,
            shard_index: 1,
            is_coordinator: false,
        },
        Arc::clone(&backend),
        transport.scoped(w2),
        Arc::clone(&chain),
    );
    let worker3 = Worker::new(
        WorkerConfig {
            job_id,
            self_address: w3,
            shard_index: 2,
            is_coordinator: false,
        },
        Arc::clone(&backend),
        transport.scoped(w3),
        Arc::clone(&chain),
    );

    // Run all three concurrently.
    let h1 = tokio::spawn(worker1.run());
    let h2 = tokio::spawn(worker2.run());
    let h3 = tokio::spawn(worker3.run());

    let results = tokio::join!(h1, h2, h3);

    let outcome1 = results.0.expect("w1 join").expect("w1 run");
    let outcome2 = results.1.expect("w2 join").expect("w2 run");
    let outcome3 = results.2.expect("w3 join").expect("w3 run");

    assert_eq!(outcome1, WorkerOutcome::Finalized, "w1 outcome");
    assert_eq!(outcome2, WorkerOutcome::Finalized, "w2 outcome");
    assert_eq!(outcome3, WorkerOutcome::Finalized, "w3 outcome");

    // Verify the chain's view: epoch roots posted, state finalized.
    let snap = chain.snapshot(job_id).await.expect("snapshot");
    assert_eq!(snap.state, JobChainState::Finalized);
    assert_eq!(snap.epoch_roots.len(), spec.epoch_count as usize);
    for epoch in 0..spec.epoch_count {
        let root = snap
            .epoch_roots
            .get(&epoch)
            .expect("epoch root should exist");
        assert_ne!(*root, H256::zero(), "epoch {} root must be non-zero", epoch);
    }

    // Sanity: the two epoch roots differ. The DeterministicTinyModel
    // chains prev_weights through each step, so every epoch's
    // commits differ and the aggregated roots must too.
    let root0 = snap.epoch_roots.get(&0).copied().unwrap();
    let root1 = snap.epoch_roots.get(&1).copied().unwrap();
    assert_ne!(root0, root1, "epoch roots should differ");
}
