//! CM-08 pipeline-parallel inference mode for citrate-training-worker.
//!
//! A pipeline-mode worker owns ONE stage of a frontier-scale model.
//! Requests flow through stages sequentially: stage 0 receives the
//! input, runs its layer slice, forwards the activation to stage 1,
//! ..., the last stage emits generated tokens back to the requester.
//!
//! This module layers on the shared trait vocabulary from the
//! training side:
//!
//! - `Transport` — carries `PipelineActivation` messages between
//!   sequential stages (new variant added in `transport.rs`).
//! - `ChainClient` surface extended via `PipelineChainClient` trait
//!   — pipeline contract has a different shape (per-request
//!   progress counter, stage ownership per job) than training's
//!   per-epoch commits.
//! - `ModelBackend` trait is REUSED with a new method
//!   `pipeline_stage_forward` that takes an activation + stage
//!   index and returns the next activation.
//!
//! # S0 scope (this slice)
//!
//! - `PipelineWorker` state machine for one stage, driven in-process
//!   via `InProcessTransport` + `MockPipelineChainClient`
//! - 4-stage happy-path integration test
//! - Deterministic activation chaining via `DeterministicTinyModel`
//!   extended with `pipeline_stage_forward`
//!
//! # Deferred to S1+
//!
//! - libp2p transport for cross-machine activation forwarding
//! - Real GPU tensor execution per stage
//! - Stage-fault detection + reassignment logic in the worker
//!   (chain-level support is in `ComputePoolPipeline.reassignStage`
//!   from WP-08.1 already)
//! - TEE attestation integration (fetch + submit to
//!   TEEAttestationRegistry; local cryptographic TEE evidence
//!   production)

use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info};

use crate::transport::{Transport, WorkerMessage};
use crate::types::{B256, WorkerAddress};

// ── Types ───────────────────────────────────────────────────────

pub type PipelineJobId = u64;
pub type PipelineRequestId = u64;
pub type StageIndex = u32;

/// Training-worker-side view of a pipeline job. Mirrors
/// `ComputePoolPipeline.Job` but with only the fields the worker
/// needs.
#[derive(Clone, Debug)]
pub struct PipelineJobSpec {
    pub stage_count: u32,
    pub payment_per_request: u128,
    pub per_stake_per_stage: u128,
    pub model_hash: B256,
}

/// A specific worker's role within a pipeline job.
#[derive(Clone, Debug)]
pub struct StageRole {
    pub job_id: PipelineJobId,
    pub stage_index: StageIndex,
    pub self_address: WorkerAddress,
    pub total_stages: u32,
}

impl StageRole {
    pub fn is_first_stage(&self) -> bool {
        self.stage_index == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.stage_index + 1 == self.total_stages
    }

    pub fn next_stage(&self) -> Option<StageIndex> {
        if self.is_last_stage() {
            None
        } else {
            Some(self.stage_index + 1)
        }
    }
}

/// Synchronous request view a worker sees.
#[derive(Clone, Debug)]
pub struct PipelineRequestSnapshot {
    pub request_id: PipelineRequestId,
    pub job_id: PipelineJobId,
    pub requester: WorkerAddress,
    pub progress: StageIndex,
    pub state: PipelineRequestState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineRequestState {
    InFlight,
    Completed,
    Failed,
}

// ── Pipeline chain client ───────────────────────────────────────

#[derive(Error, Debug)]
pub enum PipelineChainError {
    #[error("unknown job: {0}")]
    UnknownJob(PipelineJobId),
    #[error("unknown request: {0}")]
    UnknownRequest(PipelineRequestId),
    #[error("wrong state: {0}")]
    WrongState(String),
    #[error("caller is not stage owner for stage {stage}")]
    NotStageOwner { stage: StageIndex },
    #[error("stage {0} unowned (faulted)")]
    StageUnowned(StageIndex),
}

#[async_trait]
pub trait PipelineChainClient: Send + Sync {
    /// Current snapshot of a request's on-chain state.
    async fn request_snapshot(
        &self,
        request_id: PipelineRequestId,
    ) -> Result<PipelineRequestSnapshot, PipelineChainError>;

    /// Read the current owner of a specific stage. None if faulted.
    async fn stage_owner(
        &self,
        job_id: PipelineJobId,
        stage: StageIndex,
    ) -> Result<Option<WorkerAddress>, PipelineChainError>;

    /// Advance the request — caller must be the current stage
    /// owner for the request's `progress` stage. Increments
    /// `progress` and credits `paymentPerStage` to the caller.
    async fn advance_request(
        &self,
        request_id: PipelineRequestId,
        caller: WorkerAddress,
    ) -> Result<PipelineRequestState, PipelineChainError>;
}

// ── Mock pipeline chain client ──────────────────────────────────

/// In-memory PipelineChainClient mirroring `ComputePoolPipeline`'s
/// progression semantics. Tests that pass here catch the same
/// state-machine bugs a live contract would.
pub struct MockPipelineChainClient {
    inner: Arc<Mutex<MockPipelineState>>,
}

struct MockPipelineState {
    jobs: HashMap<PipelineJobId, MockPipelineJob>,
    requests: HashMap<PipelineRequestId, MockPipelineRequest>,
    next_job_id: PipelineJobId,
    next_request_id: PipelineRequestId,
}

struct MockPipelineJob {
    spec: PipelineJobSpec,
    stage_owners: HashMap<StageIndex, WorkerAddress>,
}

struct MockPipelineRequest {
    job_id: PipelineJobId,
    requester: WorkerAddress,
    progress: StageIndex,
    state: PipelineRequestState,
    payment_earned: HashMap<WorkerAddress, u128>,
}

impl MockPipelineChainClient {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(Mutex::new(MockPipelineState {
                jobs: HashMap::new(),
                requests: HashMap::new(),
                next_job_id: 0,
                next_request_id: 0,
            })),
        })
    }

    /// Test-side helper: create a job + assign stage owners. In
    /// production these land as a sequence of on-chain txs
    /// (createPipelineJob + N × assignStage + activateJob).
    pub fn create_active_job(
        &self,
        spec: PipelineJobSpec,
        stage_owners: Vec<WorkerAddress>,
    ) -> PipelineJobId {
        assert_eq!(
            stage_owners.len() as u32,
            spec.stage_count,
            "stage_owners count must match spec.stage_count"
        );
        let mut state = self.inner.lock();
        let id = state.next_job_id;
        state.next_job_id += 1;
        let owner_map: HashMap<StageIndex, WorkerAddress> = stage_owners
            .into_iter()
            .enumerate()
            .map(|(i, w)| (i as u32, w))
            .collect();
        state.jobs.insert(
            id,
            MockPipelineJob {
                spec,
                stage_owners: owner_map,
            },
        );
        id
    }

    /// Submit a new request against a job. Returns the request id.
    pub fn submit_request(
        &self,
        job_id: PipelineJobId,
        requester: WorkerAddress,
    ) -> Result<PipelineRequestId, PipelineChainError> {
        let mut state = self.inner.lock();
        if !state.jobs.contains_key(&job_id) {
            return Err(PipelineChainError::UnknownJob(job_id));
        }
        let id = state.next_request_id;
        state.next_request_id += 1;
        state.requests.insert(
            id,
            MockPipelineRequest {
                job_id,
                requester,
                progress: 0,
                state: PipelineRequestState::InFlight,
                payment_earned: HashMap::new(),
            },
        );
        Ok(id)
    }

    /// Test-side helper: how much did `worker` earn serving this
    /// request?
    pub fn payment_earned(
        &self,
        request_id: PipelineRequestId,
        worker: WorkerAddress,
    ) -> u128 {
        let state = self.inner.lock();
        state
            .requests
            .get(&request_id)
            .and_then(|r| r.payment_earned.get(&worker).copied())
            .unwrap_or(0)
    }
}

impl Default for MockPipelineChainClient {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockPipelineState {
                jobs: HashMap::new(),
                requests: HashMap::new(),
                next_job_id: 0,
                next_request_id: 0,
            })),
        }
    }
}

#[async_trait]
impl PipelineChainClient for MockPipelineChainClient {
    async fn request_snapshot(
        &self,
        request_id: PipelineRequestId,
    ) -> Result<PipelineRequestSnapshot, PipelineChainError> {
        let state = self.inner.lock();
        let req = state
            .requests
            .get(&request_id)
            .ok_or(PipelineChainError::UnknownRequest(request_id))?;
        Ok(PipelineRequestSnapshot {
            request_id,
            job_id: req.job_id,
            requester: req.requester,
            progress: req.progress,
            state: req.state.clone(),
        })
    }

    async fn stage_owner(
        &self,
        job_id: PipelineJobId,
        stage: StageIndex,
    ) -> Result<Option<WorkerAddress>, PipelineChainError> {
        let state = self.inner.lock();
        let job = state
            .jobs
            .get(&job_id)
            .ok_or(PipelineChainError::UnknownJob(job_id))?;
        Ok(job.stage_owners.get(&stage).copied())
    }

    async fn advance_request(
        &self,
        request_id: PipelineRequestId,
        caller: WorkerAddress,
    ) -> Result<PipelineRequestState, PipelineChainError> {
        let mut state = self.inner.lock();
        let req = state
            .requests
            .get_mut(&request_id)
            .ok_or(PipelineChainError::UnknownRequest(request_id))?;
        if req.state != PipelineRequestState::InFlight {
            return Err(PipelineChainError::WrongState("not in-flight".into()));
        }
        let job_id = req.job_id;
        let progress = req.progress;
        let job = state
            .jobs
            .get(&job_id)
            .ok_or(PipelineChainError::UnknownJob(job_id))?;
        let owner = job
            .stage_owners
            .get(&progress)
            .copied()
            .ok_or(PipelineChainError::StageUnowned(progress))?;
        if owner != caller {
            return Err(PipelineChainError::NotStageOwner { stage: progress });
        }
        let stage_share = job.spec.payment_per_request / u128::from(job.spec.stage_count);
        let total_stages = job.spec.stage_count;
        let req = state.requests.get_mut(&request_id).expect("checked above");
        *req.payment_earned.entry(caller).or_insert(0) += stage_share;
        req.progress += 1;
        if req.progress == total_stages {
            req.state = PipelineRequestState::Completed;
        }
        Ok(req.state.clone())
    }
}

// ── Pipeline worker ────────────────────────────────────────────

/// Processes activations for ONE stage of a pipeline. One worker
/// per stage per job.
pub struct PipelineWorker<T, C>
where
    T: Transport + 'static,
    C: PipelineChainClient + 'static,
{
    pub role: StageRole,
    pub transport: Arc<T>,
    pub chain: Arc<C>,
}

impl<T, C> PipelineWorker<T, C>
where
    T: Transport + 'static,
    C: PipelineChainClient + 'static,
{
    pub fn new(role: StageRole, transport: Arc<T>, chain: Arc<C>) -> Self {
        Self {
            role,
            transport,
            chain,
        }
    }

    /// Serve a single request's activation through this stage.
    ///
    /// For stage 0: the `input` parameter is the request prompt
    /// bytes; the worker runs the first stage, forwards to stage 1,
    /// then calls `advance_request`.
    ///
    /// For intermediate stages: we WAIT for an incoming activation
    /// on the transport, run our stage, forward to the next stage,
    /// then advance.
    ///
    /// For the last stage: we wait for incoming activation, run,
    /// emit the result (S0 just returns the final payload), advance.
    ///
    /// Returns the final activation at the terminal stage (or
    /// `None` for non-terminal stages).
    pub async fn serve_request(
        &self,
        request_id: PipelineRequestId,
        first_stage_input: Option<Vec<u8>>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        info!(
            worker = %self.role.self_address,
            stage = self.role.stage_index,
            request = request_id,
            "pipeline worker starting stage"
        );

        // Receive or supply the input activation.
        let incoming: Vec<u8> = if self.role.is_first_stage() {
            first_stage_input.ok_or_else(|| {
                anyhow::anyhow!("stage 0 requires first_stage_input")
            })?
        } else {
            loop {
                let msg = self
                    .transport
                    .recv(self.role.self_address)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("transport closed"))?;
                if let WorkerMessage::PipelineActivation {
                    request_id: rid,
                    to_stage,
                    payload,
                    ..
                } = msg
                {
                    if rid == request_id && to_stage == self.role.stage_index {
                        break payload;
                    }
                    // Discard mismatched messages (wrong request or wrong stage).
                    debug!(
                        got_request = rid,
                        got_stage = to_stage,
                        expected_request = request_id,
                        expected_stage = self.role.stage_index,
                        "skipping mismatched pipeline message"
                    );
                }
            }
        };

        // Run this stage's computation. S0 uses a deterministic
        // chaining: output = keccak(stage_index || input). Real
        // GPU execution lands in S2 as a ModelBackend impl.
        let output = pipeline_stage_forward(
            self.role.stage_index,
            self.role.total_stages,
            &incoming,
        );

        // Forward to the next stage, OR return the final output.
        let final_output = if let Some(next) = self.role.next_stage() {
            self.transport
                .broadcast(WorkerMessage::PipelineActivation {
                    request_id,
                    from_stage: self.role.stage_index,
                    to_stage: next,
                    payload: output.clone(),
                })
                .await?;
            None
        } else {
            Some(output.clone())
        };

        // Advance on-chain: claims this stage's payment share +
        // bumps request progress.
        self.chain
            .advance_request(request_id, self.role.self_address)
            .await
            .map_err(|e| anyhow::anyhow!("advance_request: {}", e))?;

        info!(
            worker = %self.role.self_address,
            stage = self.role.stage_index,
            request = request_id,
            "pipeline stage served"
        );
        Ok(final_output)
    }
}

/// Deterministic S0 stage-forward. Hashes (stage_index || input)
/// with keccak256 and returns the 32-byte output as the next-stage
/// activation. This gives the pipeline a non-trivial data
/// dependency between stages while staying reproducible across
/// workers — a challenger can recompute.
///
/// S2 replaces this with a real ModelBackend call holding the
/// stage's model-slice weights.
pub fn pipeline_stage_forward(
    stage_index: StageIndex,
    _total_stages: u32,
    input: &[u8],
) -> Vec<u8> {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(stage_index.to_be_bytes());
    hasher.update(input);
    let out = hasher.finalize();
    out.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethereum_types::Address;

    #[test]
    fn stage_role_boundaries() {
        let first = StageRole {
            job_id: 0,
            stage_index: 0,
            self_address: Address::repeat_byte(1),
            total_stages: 4,
        };
        let last = StageRole {
            job_id: 0,
            stage_index: 3,
            self_address: Address::repeat_byte(4),
            total_stages: 4,
        };
        assert!(first.is_first_stage());
        assert!(!first.is_last_stage());
        assert_eq!(first.next_stage(), Some(1));

        assert!(!last.is_first_stage());
        assert!(last.is_last_stage());
        assert_eq!(last.next_stage(), None);
    }

    #[test]
    fn pipeline_stage_forward_is_deterministic() {
        let input = b"hello world";
        let a = pipeline_stage_forward(2, 4, input);
        let b = pipeline_stage_forward(2, 4, input);
        assert_eq!(a, b);
    }

    #[test]
    fn pipeline_stage_forward_differs_per_stage() {
        let input = b"hello";
        let s0 = pipeline_stage_forward(0, 4, input);
        let s1 = pipeline_stage_forward(1, 4, input);
        assert_ne!(s0, s1);
    }

    #[tokio::test]
    async fn mock_chain_full_lifecycle() {
        let chain = MockPipelineChainClient::new();
        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let w3 = Address::repeat_byte(3);
        let w4 = Address::repeat_byte(4);
        let requester = Address::repeat_byte(0xAA);

        let spec = PipelineJobSpec {
            stage_count: 4,
            payment_per_request: 4_000_000_000_000_000_000u128, // 4 ether
            per_stake_per_stage: 2_000_000_000_000_000_000u128,
            model_hash: B256::repeat_byte(0x11),
        };
        let job_id = chain.create_active_job(spec, vec![w1, w2, w3, w4]);

        let req_id = chain.submit_request(job_id, requester).expect("submit");
        let snap = chain.request_snapshot(req_id).await.unwrap();
        assert_eq!(snap.progress, 0);
        assert_eq!(snap.state, PipelineRequestState::InFlight);

        // Each stage advances in order.
        for (stage, worker) in [(0u32, w1), (1, w2), (2, w3), (3, w4)] {
            let state = chain.advance_request(req_id, worker).await.unwrap();
            if stage < 3 {
                assert_eq!(state, PipelineRequestState::InFlight);
            } else {
                assert_eq!(state, PipelineRequestState::Completed);
            }
        }

        // Each worker earned payment / 4 = 1 ether.
        for w in [w1, w2, w3, w4] {
            assert_eq!(chain.payment_earned(req_id, w), 1_000_000_000_000_000_000u128);
        }
    }

    #[tokio::test]
    async fn mock_chain_rejects_wrong_stage_caller() {
        let chain = MockPipelineChainClient::new();
        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let requester = Address::repeat_byte(0xAA);
        let spec = PipelineJobSpec {
            stage_count: 2,
            payment_per_request: 2_000_000_000_000_000_000u128,
            per_stake_per_stage: 1_000_000_000_000_000_000u128,
            model_hash: B256::repeat_byte(0x11),
        };
        let job_id = chain.create_active_job(spec, vec![w1, w2]);
        let req_id = chain.submit_request(job_id, requester).unwrap();

        // w2 tries to advance while w1 holds stage 0.
        let err = chain.advance_request(req_id, w2).await.unwrap_err();
        assert!(matches!(err, PipelineChainError::NotStageOwner { stage: 0 }));
    }
}
