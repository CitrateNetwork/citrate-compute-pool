//! A non-coordinator worker waits for the coordinator's epoch root on chain.
//! That wait must end: a chain whose snapshot never shows the root must make
//! `run()` return an error, not poll forever.
//!
//! The deadline is minutes long, so the test runs on tokio's paused clock: time
//! advances only when every task is idle, and the outer timeout is far beyond
//! the deadline. With the bound in place `run()` errors at the deadline; without
//! it the outer timeout fires instead.

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
        min_workers: 1,
        max_workers: 1,
        challenge_window_blocks: 1,
        per_epoch_budget: 30_000_000_000_000_000_000u128,
        per_worker_stake: 10_000_000_000_000_000_000u128,
    }
}

#[tokio::test(start_paused = true)]
async fn a_missing_epoch_root_ends_the_wait_with_an_error() {
    let chain = MockChainClient::new();
    let transport = InProcessTransport::new();
    let me = Address::repeat_byte(0xC1);
    let job_id = chain.create_job(spec());
    chain
        .join_training_job(job_id, me, spec().per_worker_stake)
        .await
        .expect("join");
    transport.register(me).await;
    chain.close_recruitment(job_id, me).await.expect("close");

    // Not the coordinator, and nobody else is: the root never appears.
    let worker = Worker::new(
        WorkerConfig {
            job_id,
            self_address: me,
            shard_index: 0,
            is_coordinator: false,
        },
        Arc::new(DeterministicTinyModel::new()),
        transport.scoped(me),
        Arc::clone(&chain),
    );

    let out = tokio::time::timeout(Duration::from_secs(24 * 3600), worker.run())
        .await
        .expect("the epoch-root wait must give up on its own deadline");
    let err = out.expect_err("a root that never appears must fail the run");
    assert!(
        err.to_string().contains("epoch root did not appear"),
        "the error must name the missing epoch root, got: {err}"
    );
}
