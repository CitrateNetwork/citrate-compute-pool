//! `ChainClient` trait + `MockChainClient` for S0.
//!
//! Wraps the subset of ComputePoolTraining calls a worker daemon
//! needs. For S0 the mock reproduces the on-chain state machine in
//! memory — joins, epoch commits, finalize. It enforces the same
//! invariants the Solidity contract does (epoch monotonicity,
//! coordinator-only commitEpoch, challenge-window-gated finalize)
//! so tests fail the same way a live chain would.
//!
//! S1+ real `HttpChainClient` implementation will speak JSON-RPC to
//! a Citrate node and encode the same calldata the SDK does.

use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

use crate::types::{B256, EpochIndex, JobId, TrainingJobSpec, WorkerAddress};

#[derive(Error, Debug)]
pub enum ChainError {
    #[error("unknown job: {0}")]
    UnknownJob(JobId),
    #[error("wrong state for operation: {0}")]
    WrongState(String),
    #[error("worker already joined")]
    AlreadyJoined,
    #[error("pool full")]
    PoolFull,
    #[error("stake amount mismatch")]
    StakeMismatch,
    #[error("below min workers")]
    BelowMin,
    #[error("coordinator is not a joined worker")]
    CoordinatorNotJoined,
    #[error("wrong epoch; expected {expected}, got {got}")]
    WrongEpoch { expected: EpochIndex, got: EpochIndex },
    #[error("epoch already committed")]
    EpochAlreadyCommitted,
    #[error("challenge window still open")]
    ChallengeWindowOpen,
    #[error("caller is not coordinator")]
    NotCoordinator,
    #[error("job not ready to finalize")]
    NotReadyToFinalize,
    #[error("caller not joined; only joined workers can trigger reassignment")]
    CallerNotJoined,
    #[error("proposed replacement is not a joined worker")]
    ReplacementNotJoined,
    #[error("coordinator still active within timeout")]
    CoordinatorStillActive,
}

/// Summary of a training job's on-chain state as seen by a worker.
#[derive(Clone, Debug)]
pub struct JobChainSnapshot {
    pub spec: TrainingJobSpec,
    pub state: JobChainState,
    pub current_epoch: EpochIndex,
    pub coordinator: Option<WorkerAddress>,
    pub workers: Vec<WorkerAddress>,
    pub epoch_roots: HashMap<EpochIndex, B256>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobChainState {
    Recruiting,
    Training,
    Awaiting,
    Finalized,
    Aborted,
}

#[async_trait]
pub trait ChainClient: Send + Sync {
    /// Read the current snapshot of a job's on-chain state.
    async fn snapshot(&self, job_id: JobId) -> Result<JobChainSnapshot, ChainError>;

    /// Worker posts stake + joins. No-op if the caller already
    /// joined.
    async fn join_training_job(
        &self,
        job_id: JobId,
        sender: WorkerAddress,
        stake: u128,
    ) -> Result<(), ChainError>;

    /// Close recruitment and elect the initial coordinator.
    /// Callable by anyone once min_workers is met.
    async fn close_recruitment(
        &self,
        job_id: JobId,
        coordinator: WorkerAddress,
    ) -> Result<(), ChainError>;

    /// Coordinator posts the Merkle root for the current epoch.
    async fn commit_epoch(
        &self,
        job_id: JobId,
        sender: WorkerAddress,
        epoch: EpochIndex,
        root: B256,
    ) -> Result<(), ChainError>;

    /// Advance the chain's simulated block number. Tests use this
    /// to roll past the challenge window.
    async fn advance_blocks(&self, n: u64);

    /// Finalize the job. Reverts with ChallengeWindowOpen if
    /// called too early.
    async fn finalize(&self, job_id: JobId) -> Result<(), ChainError>;

    /// Reassign a stalled coordinator. Caller must be a joined
    /// worker; reassignment is only accepted if the last coordinator
    /// activity was more than `coordination_timeout` blocks ago.
    /// Triggers a liveness slash on the old coordinator. Mirrors
    /// `ComputePoolTraining.reassignCoordinator`.
    async fn reassign_coordinator(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
        new_coordinator: WorkerAddress,
    ) -> Result<(), ChainError>;
}

/// In-memory implementation of the ComputePoolTraining state
/// machine. Mirrors the Solidity contract's invariants byte-for-
/// byte (join → close → commit × E → finalize) so tests
/// exercising the worker state machine catch the same class of
/// bugs they'd catch against a live deployment.
pub struct MockChainClient {
    inner: Arc<Mutex<MockState>>,
}

struct MockState {
    jobs: HashMap<JobId, MockJob>,
    next_job_id: JobId,
    block_number: u64,
}

#[derive(Clone)]
struct MockJob {
    spec: TrainingJobSpec,
    state: JobChainState,
    current_epoch: EpochIndex,
    coordinator: Option<WorkerAddress>,
    workers: Vec<WorkerAddress>,
    joined: HashSet<WorkerAddress>,
    epoch_roots: HashMap<EpochIndex, B256>,
    all_epochs_committed_block: u64,
    last_activity_block: u64,
    liveness_slashed: HashSet<WorkerAddress>,
}

impl MockChainClient {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(Mutex::new(MockState {
                jobs: HashMap::new(),
                next_job_id: 0,
                block_number: 0,
            })),
        })
    }

    /// Create a new training job (test-side helper — no actual
    /// escrow accounting in the mock; the lifecycle invariants are
    /// what we're validating).
    pub fn create_job(&self, spec: TrainingJobSpec) -> JobId {
        let mut state = self.inner.lock();
        let id = state.next_job_id;
        state.next_job_id += 1;
        state.jobs.insert(
            id,
            MockJob {
                spec,
                state: JobChainState::Recruiting,
                current_epoch: 0,
                coordinator: None,
                workers: Vec::new(),
                joined: HashSet::new(),
                epoch_roots: HashMap::new(),
                all_epochs_committed_block: 0,
                last_activity_block: 0,
                liveness_slashed: HashSet::new(),
            },
        );
        id
    }

    /// Test-side helper: has a worker been liveness-slashed for
    /// coordinator-stall? Mirrors the on-chain WorkerInfo.stakeSlashed
    /// being non-zero after a reassign call.
    pub fn was_liveness_slashed(&self, job_id: JobId, worker: WorkerAddress) -> bool {
        self.inner
            .lock()
            .jobs
            .get(&job_id)
            .map(|j| j.liveness_slashed.contains(&worker))
            .unwrap_or(false)
    }
}

/// Coordination-timeout constant mirroring the Solidity contract's
/// COORDINATION_TIMEOUT. The mock reuses the on-chain value so tests
/// don't drift from the deployed behavior.
const COORDINATION_TIMEOUT: u64 = 100;

impl Default for MockChainClient {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockState {
                jobs: HashMap::new(),
                next_job_id: 0,
                block_number: 0,
            })),
        }
    }
}

#[async_trait]
impl ChainClient for MockChainClient {
    async fn snapshot(&self, job_id: JobId) -> Result<JobChainSnapshot, ChainError> {
        let state = self.inner.lock();
        let job = state
            .jobs
            .get(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        Ok(JobChainSnapshot {
            spec: job.spec.clone(),
            state: job.state.clone(),
            current_epoch: job.current_epoch,
            coordinator: job.coordinator,
            workers: job.workers.clone(),
            epoch_roots: job.epoch_roots.clone(),
        })
    }

    async fn join_training_job(
        &self,
        job_id: JobId,
        sender: WorkerAddress,
        stake: u128,
    ) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        let job = state
            .jobs
            .get_mut(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        if job.state != JobChainState::Recruiting {
            return Err(ChainError::WrongState("not recruiting".into()));
        }
        if job.joined.contains(&sender) {
            return Err(ChainError::AlreadyJoined);
        }
        if job.workers.len() as u32 >= job.spec.max_workers {
            return Err(ChainError::PoolFull);
        }
        if stake != job.spec.per_worker_stake {
            return Err(ChainError::StakeMismatch);
        }
        job.workers.push(sender);
        job.joined.insert(sender);
        Ok(())
    }

    async fn close_recruitment(
        &self,
        job_id: JobId,
        coordinator: WorkerAddress,
    ) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        let block = state.block_number;
        let job = state
            .jobs
            .get_mut(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        if job.state != JobChainState::Recruiting {
            return Err(ChainError::WrongState("not recruiting".into()));
        }
        if (job.workers.len() as u32) < job.spec.min_workers {
            return Err(ChainError::BelowMin);
        }
        if !job.joined.contains(&coordinator) {
            return Err(ChainError::CoordinatorNotJoined);
        }
        job.state = JobChainState::Training;
        job.coordinator = Some(coordinator);
        job.last_activity_block = block;
        Ok(())
    }

    async fn commit_epoch(
        &self,
        job_id: JobId,
        sender: WorkerAddress,
        epoch: EpochIndex,
        root: B256,
    ) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        let block = state.block_number;
        let job = state
            .jobs
            .get_mut(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        if job.state != JobChainState::Training {
            return Err(ChainError::WrongState("not training".into()));
        }
        if job.coordinator != Some(sender) {
            return Err(ChainError::NotCoordinator);
        }
        if epoch != job.current_epoch {
            return Err(ChainError::WrongEpoch {
                expected: job.current_epoch,
                got: epoch,
            });
        }
        if job.epoch_roots.contains_key(&epoch) {
            return Err(ChainError::EpochAlreadyCommitted);
        }
        job.epoch_roots.insert(epoch, root);
        job.current_epoch += 1;
        job.last_activity_block = block;
        if job.current_epoch == job.spec.epoch_count {
            job.state = JobChainState::Awaiting;
            job.all_epochs_committed_block = block;
        }
        Ok(())
    }

    async fn advance_blocks(&self, n: u64) {
        let mut state = self.inner.lock();
        state.block_number += n;
    }

    async fn finalize(&self, job_id: JobId) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        let block = state.block_number;
        let job = state
            .jobs
            .get_mut(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        if job.state != JobChainState::Awaiting {
            return Err(ChainError::NotReadyToFinalize);
        }
        let window_end = job.all_epochs_committed_block + job.spec.challenge_window_blocks as u64;
        if block < window_end {
            return Err(ChainError::ChallengeWindowOpen);
        }
        job.state = JobChainState::Finalized;
        Ok(())
    }

    async fn reassign_coordinator(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
        new_coordinator: WorkerAddress,
    ) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        let block = state.block_number;
        let job = state
            .jobs
            .get_mut(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        if job.state != JobChainState::Training {
            return Err(ChainError::WrongState("not training".into()));
        }
        if !job.joined.contains(&caller) {
            return Err(ChainError::CallerNotJoined);
        }
        if !job.joined.contains(&new_coordinator) {
            return Err(ChainError::ReplacementNotJoined);
        }
        if block <= job.last_activity_block + COORDINATION_TIMEOUT {
            return Err(ChainError::CoordinatorStillActive);
        }
        if let Some(old) = job.coordinator {
            job.liveness_slashed.insert(old);
        }
        job.coordinator = Some(new_coordinator);
        job.last_activity_block = block;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethereum_types::Address;

    fn spec() -> TrainingJobSpec {
        TrainingJobSpec {
            model_start_hash: B256::repeat_byte(0x11),
            dataset_hash: B256::repeat_byte(0x22),
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
    async fn full_lifecycle_on_mock() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());

        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let w3 = Address::repeat_byte(3);

        chain
            .join_training_job(job_id, w1, spec().per_worker_stake)
            .await
            .unwrap();
        chain
            .join_training_job(job_id, w2, spec().per_worker_stake)
            .await
            .unwrap();
        chain
            .join_training_job(job_id, w3, spec().per_worker_stake)
            .await
            .unwrap();

        chain.close_recruitment(job_id, w1).await.unwrap();

        let snap = chain.snapshot(job_id).await.unwrap();
        assert_eq!(snap.state, JobChainState::Training);
        assert_eq!(snap.coordinator, Some(w1));

        chain
            .commit_epoch(job_id, w1, 0, B256::repeat_byte(0xEE))
            .await
            .unwrap();
        chain
            .commit_epoch(job_id, w1, 1, B256::repeat_byte(0xFF))
            .await
            .unwrap();

        let snap = chain.snapshot(job_id).await.unwrap();
        assert_eq!(snap.state, JobChainState::Awaiting);

        // Too-early finalize is rejected.
        let err = chain.finalize(job_id).await.unwrap_err();
        assert!(matches!(err, ChainError::ChallengeWindowOpen));

        chain.advance_blocks(11).await;
        chain.finalize(job_id).await.unwrap();

        let snap = chain.snapshot(job_id).await.unwrap();
        assert_eq!(snap.state, JobChainState::Finalized);
    }

    #[tokio::test]
    async fn non_coordinator_cannot_commit_epoch() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());

        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let w3 = Address::repeat_byte(3);

        for w in [w1, w2, w3] {
            chain
                .join_training_job(job_id, w, spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain.close_recruitment(job_id, w1).await.unwrap();

        let err = chain
            .commit_epoch(job_id, w2, 0, B256::repeat_byte(0xEE))
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::NotCoordinator));
    }

    #[tokio::test]
    async fn reassign_after_timeout_swaps_coordinator() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());
        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let w3 = Address::repeat_byte(3);
        for w in [w1, w2, w3] {
            chain
                .join_training_job(job_id, w, spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain.close_recruitment(job_id, w1).await.unwrap();

        // Too early.
        chain.advance_blocks(50).await;
        let err = chain.reassign_coordinator(job_id, w2, w3).await.unwrap_err();
        assert!(matches!(err, ChainError::CoordinatorStillActive));

        // Past timeout.
        chain.advance_blocks(60).await;
        chain.reassign_coordinator(job_id, w2, w3).await.unwrap();

        let snap = chain.snapshot(job_id).await.unwrap();
        assert_eq!(snap.coordinator, Some(w3), "coordinator swapped");
        assert!(chain.was_liveness_slashed(job_id, w1));
    }

    #[tokio::test]
    async fn reassign_rejects_non_member_caller() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());
        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let w3 = Address::repeat_byte(3);
        for w in [w1, w2, w3] {
            chain
                .join_training_job(job_id, w, spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain.close_recruitment(job_id, w1).await.unwrap();
        chain.advance_blocks(200).await;

        let outsider = Address::repeat_byte(0xEE);
        let err = chain
            .reassign_coordinator(job_id, outsider, w3)
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::CallerNotJoined));
    }

    #[tokio::test]
    async fn epoch_monotonicity_enforced() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());
        let w1 = Address::repeat_byte(1);
        let w2 = Address::repeat_byte(2);
        let w3 = Address::repeat_byte(3);
        for w in [w1, w2, w3] {
            chain
                .join_training_job(job_id, w, spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain.close_recruitment(job_id, w1).await.unwrap();

        // Skipping to epoch 1 before 0 reverts.
        let err = chain
            .commit_epoch(job_id, w1, 1, B256::repeat_byte(0xEE))
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::WrongEpoch { .. }));
    }
}
