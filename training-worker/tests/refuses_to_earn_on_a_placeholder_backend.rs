//! A worker must REFUSE to earn against a live settlement chain while its
//! backend is a placeholder.
//!
//! ## Why this test exists
//!
//! `ComputePoolTraining` pays workers per epoch (`EpochPaymentReleased`). A
//! worker earns that money by committing a Merkle root of its step commitments.
//! Nothing in the protocol can tell whether those commitments came from real
//! training — they are just hashes of gradient tensors.
//!
//! Today NO backend in this crate trains the job the chain describes:
//!
//!   - `ModelBackend::load_starting_weights` returns the hash it was handed;
//!     actually fetching those weights is deferred to S2/S3 (see the trait doc).
//!   - `spec.dataset_hash` is decoded from chain into `TrainingJobSpec` and then
//!     consumed by nothing — grep it.
//!   - `CandleBackend` trains a locally-initialised linear layer on
//!     `synthetic_input(epoch, step, shard)` — arithmetic, not data — with
//!     `loss = sum(y)`, which its own comment calls a placeholder. Its module
//!     doc is explicit: "The point of this reference impl isn't ML accuracy."
//!   - `DeterministicTinyModel` derives gradients from a keccak hash, "without
//!     any real training" in its own words.
//!
//! Each is a fine S0/S1 harness. What must never happen is one of them meeting
//! a REAL chain: the daemon would commit legitimate-looking roots, collect real
//! SALT per epoch, and be indistinguishable on-chain from a worker that did the
//! job. That is Rule 1 at money scale — and it fails silently, which is why it
//! needs a hard gate rather than a note in a doc.
//!
//! The gate is two capability bits that must BOTH be honest:
//!   - `ModelBackend::honors_job_spec()` — "I load `model_start_hash` and train
//!     on `dataset_hash`." Defaults to FALSE, so a new backend is untrusted
//!     until it deliberately claims otherwise.
//!   - `ChainClient::is_live_settlement()` — "commits I make here pay real
//!     money." Defaults to FALSE, so test harnesses are unaffected.

use std::sync::Arc;

use async_trait::async_trait;
use citrate_training_worker::backend::StepResult;
use citrate_training_worker::types::{EpochIndex, PrevWeightsHash, StepIndex, WeightsHash};
use citrate_training_worker::{
    ChainClient, DeterministicTinyModel, InProcessTransport, MockChainClient, ModelBackend,
    TrainingJobSpec, Transport, Worker, WorkerConfig,
};
use ethereum_types::{Address, H256};

fn make_spec() -> TrainingJobSpec {
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

/// A backend that DOES honour the on-chain spec. No such backend exists in the
/// crate yet — that is the point — so the test declares one, to prove the gate
/// OPENS for a real implementation instead of being a blanket refusal.
struct HonestBackend(DeterministicTinyModel);

#[async_trait]
impl ModelBackend for HonestBackend {
    fn honors_job_spec(&self) -> bool {
        true
    }
    async fn load_starting_weights(&self, h: WeightsHash) -> anyhow::Result<WeightsHash> {
        self.0.load_starting_weights(h).await
    }
    async fn forward_backward(
        &self,
        prev: PrevWeightsHash,
        epoch: EpochIndex,
        step: StepIndex,
        shard: u32,
    ) -> anyhow::Result<StepResult> {
        self.0.forward_backward(prev, epoch, step, shard).await
    }
}

/// Build a single-worker job that has already closed recruitment, so `run()`
/// reaches the training loop (and therefore the gate) immediately.
async fn recruited(
    chain: &Arc<MockChainClient>,
    transport: &Arc<InProcessTransport>,
    me: Address,
) -> u64 {
    let spec = make_spec();
    let job_id = chain.create_job(spec.clone());
    chain
        .join_training_job(job_id, me, spec.per_worker_stake)
        .await
        .expect("join");
    transport.register(me).await;
    chain.close_recruitment(job_id, me).await.expect("close");
    job_id
}

/// THE GUARD: a placeholder backend + a chain that pays real money = refuse.
#[tokio::test]
async fn refuses_to_run_a_placeholder_backend_against_live_settlement() {
    let chain = MockChainClient::new_live_settlement();
    let transport = InProcessTransport::new();
    let me = Address::repeat_byte(0xA1);
    let job_id = recruited(&chain, &transport, me).await;

    let worker = Worker::new(
        WorkerConfig {
            job_id,
            self_address: me,
            shard_index: 0,
            is_coordinator: true,
        },
        Arc::new(DeterministicTinyModel::new()),
        transport.scoped(me),
        Arc::clone(&chain),
    );

    let err = worker
        .run()
        .await
        .expect_err("a placeholder backend must NOT be allowed to earn on a live chain");
    let msg = err.to_string();
    assert!(
        msg.contains("honors_job_spec"),
        "the refusal must name the capability that is missing so an operator \
         knows what to fix; got: {msg}"
    );
}

/// The gate is not an unconditional refusal — a backend that genuinely honours
/// the spec is allowed through to the lifecycle.
#[tokio::test]
async fn a_spec_honouring_backend_is_allowed_on_live_settlement() {
    let chain = MockChainClient::new_live_settlement();
    let transport = InProcessTransport::new();
    let me = Address::repeat_byte(0xA2);
    let job_id = recruited(&chain, &transport, me).await;

    let worker = Worker::new(
        WorkerConfig {
            job_id,
            self_address: me,
            shard_index: 0,
            is_coordinator: true,
        },
        Arc::new(HonestBackend(DeterministicTinyModel::new())),
        transport.scoped(me),
        Arc::clone(&chain),
    );

    if let Err(e) = worker.run().await {
        assert!(
            !e.to_string().contains("honors_job_spec"),
            "a spec-honouring backend must not be refused by the capability gate: {e}"
        );
    }
}

/// Test harnesses are unaffected: the same placeholder backend runs fine
/// against a mock chain, because nothing there pays. Without this the gate
/// would have broken every existing S0 lifecycle test — and the temptation
/// would be to weaken the gate rather than keep it honest.
#[tokio::test]
async fn a_placeholder_backend_still_runs_against_a_mock_chain() {
    let chain = MockChainClient::new();
    let transport = InProcessTransport::new();
    let me = Address::repeat_byte(0xA3);
    assert!(
        !chain.is_live_settlement(),
        "the default mock chain must not claim live settlement"
    );
    let job_id = recruited(&chain, &transport, me).await;

    let worker = Worker::new(
        WorkerConfig {
            job_id,
            self_address: me,
            shard_index: 0,
            is_coordinator: true,
        },
        Arc::new(DeterministicTinyModel::new()),
        transport.scoped(me),
        Arc::clone(&chain),
    );

    if let Err(e) = worker.run().await {
        assert!(
            !e.to_string().contains("honors_job_spec"),
            "the capability gate must not fire on a non-settling test chain: {e}"
        );
    }
}

/// The DEFAULTS are the load-bearing part: a backend or chain client added
/// later inherits the safe answer without its author having to know this gate
/// exists. If either default flips, the gate silently stops protecting anything.
#[tokio::test]
async fn the_capability_defaults_fail_safe() {
    assert!(
        !DeterministicTinyModel::new().honors_job_spec(),
        "a placeholder backend must not claim to honour the job spec"
    );
    assert!(
        !MockChainClient::new().is_live_settlement(),
        "a mock chain must not claim live settlement"
    );
}
