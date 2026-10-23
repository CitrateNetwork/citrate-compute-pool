//! The coordinator collects step commits from every worker before it finalizes an
//! epoch. A peer that joined but never sends must not wedge that drain: each
//! receive has its own timeout, so `run()` returns an error instead of waiting
//! forever.
//!
//! The per-message timeout is a minute, so the test runs on tokio's paused clock
//! (time advances only while every task is idle) with an outer timeout far beyond
//! it. With the bound in place `run()` errors on its own; without it the outer
//! timeout fires.

use std::sync::Arc;
use std::time::Duration;

use citrate_training_worker::{
    ChainClient, DeterministicTinyModel, InProcessTransport, MockChainClient, TrainingJobSpec,
    Transport, Worker, WorkerConfig,
};
use ethereum_types::{Address, H256};

fn spec() -> TrainingJobSpec {
    TrainingJobSpec {
        model_start_hash: H256::repeat_byte(0x11),
        dataset_hash: H256::repeat_byte(0x22),
        epoch_count: 1,
        steps_per_epoch: 1,
        min_workers: 2,
        max_workers: 2,
        challenge_window_blocks: 1,
        per_epoch_budget: 30_000_000_000_000_000_000u128,
        per_worker_stake: 10_000_000_000_000_000_000u128,
    }
}

#[tokio::test(start_paused = true)]
async fn a_silent_peer_ends_the_coordinators_commit_drain_with_an_error() {
    let chain = MockChainClient::new();
    let transport = InProcessTransport::new();
    let coord = Address::repeat_byte(0xC1);
    let silent = Address::repeat_byte(0xC2);
    let job_id = chain.create_job(spec());
    for w in [coord, silent] {
        chain
            .join_training_job(job_id, w, spec().per_worker_stake)
            .await
            .expect("join");
        transport.register(w).await;
    }
    chain.close_recruitment(job_id, coord).await.expect("close");

    // Only the coordinator runs; the second member joined and then went silent.
    let worker = Worker::new(
        WorkerConfig {
            job_id,
            self_address: coord,
            shard_index: 0,
            is_coordinator: true,
        },
        Arc::new(DeterministicTinyModel::new()),
        transport.scoped(coord),
        Arc::clone(&chain),
    );

    let out = tokio::time::timeout(Duration::from_secs(24 * 3600), worker.run())
        .await
        .expect("the commit drain must give up on its own per-message timeout");
    let err = out.expect_err("a peer that never sends must fail the epoch");
    assert!(
        err.to_string()
            .contains("timed out waiting for step commits"),
        "the error must name the stalled commit drain, got: {err}"
    );
}
