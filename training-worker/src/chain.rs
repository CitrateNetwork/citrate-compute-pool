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

use crate::types::{EpochIndex, JobId, TrainingJobSpec, WorkerAddress, B256};

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
    WrongEpoch {
        expected: EpochIndex,
        got: EpochIndex,
    },
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
    #[error("caller is neither the job requester nor governance")]
    NotRequesterOrGovernance,
    #[error("job not stalled; STALL_EXPIRY_BLOCKS since last activity not elapsed")]
    NotStalled,
    #[error("self-challenge not allowed")]
    SelfChallenge,
    #[error("target not joined")]
    TargetNotJoined,
    #[error("epoch not committed yet")]
    EpochNotCommitted,
    #[error("merkle proof does not verify against epoch root")]
    BadMerkleProof,
    #[error("challenge already active for this (epoch, step, target)")]
    ChallengeAlreadyActive,
    #[error("challenge not in voting state")]
    ChallengeNotVoting,
    #[error("voter not on committee")]
    VoterNotOnCommittee,
    #[error("voter already cast a vote on this challenge")]
    AlreadyVoted,
    /// RM-G2.4 / audit F-4 — mirror of `commitEpoch`'s
    /// `require(root != bytes32(0), "ComputePoolTraining: zero root")`.
    #[error("zero epoch root rejected (parity with on-chain commitEpoch)")]
    ZeroEpochRoot,
    /// RM-G2.4 / audit F-4 — mirror of `challengeStep`'s
    /// `require(step < job.stepsPerEpoch, ...)`.
    #[error("challenge step out of range")]
    StepOutOfRange,
    /// RM-G2.4 / audit F-4 — mirror of `challengeStep`'s
    /// `require(epoch < job.epochCount, ...)`.
    #[error("challenge epoch out of range")]
    ChallengeEpochOutOfRange,
    /// RM-G2.4 / audit F-4 — mirror of `challengeStep`'s
    /// `require(msg.value == CHALLENGE_BOND, ...)`.
    #[error("challenge bond must equal CHALLENGE_BOND constant")]
    WrongChallengeBond,
}

/// RM-G2.4 / audit F-4: must match `ComputePoolTraining.CHALLENGE_BOND`.
/// The on-chain constant is `1 ether`; in workspace fixed-point that's
/// 1e18 wei, which fits in `u128`.
pub const CHALLENGE_BOND: u128 = 1_000_000_000_000_000_000;

/// `ComputePoolTraining.STALL_EXPIRY_BLOCKS`: blocks of total inactivity after
/// which the requester or a joined worker may expire a Training job.
pub const STALL_EXPIRY_BLOCKS: u64 = 50_400;

/// Requester recorded by [`MockChainClient::create_job`].
pub const MOCK_REQUESTER: WorkerAddress = WorkerAddress::repeat_byte(0xA0);
/// Governance address of the mock contract.
pub const MOCK_GOVERNANCE: WorkerAddress = WorkerAddress::repeat_byte(0x60);

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
    /// Do commits made through this client SETTLE IN REAL MONEY?
    ///
    /// True for a client bound to a real `ComputePoolTraining` deployment,
    /// where a committed epoch releases real SALT via `EpochPaymentReleased`.
    ///
    /// **Defaults to `false`** so mocks and in-process harnesses are
    /// unaffected without having to know this exists. `HttpChainClient`
    /// overrides it to `true`.
    ///
    /// Paired with [`ModelBackend::honors_job_spec`]: a worker refuses to run
    /// when this is `true` and the backend cannot honour the job spec, which
    /// is the combination that would earn real money for placeholder work.
    fn is_live_settlement(&self) -> bool {
        false
    }

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

    /// Reassign a stalled coordinator. Mirrors
    /// `ComputePoolTraining.reassignCoordinator`: the caller must be the
    /// job's requester or governance (a worker cannot appoint the
    /// coordinator), the replacement must be a joined worker, and the
    /// last activity must be more than COORDINATION_TIMEOUT blocks ago.
    /// The old coordinator is liveness-slashed only when governance
    /// makes the call. Workers use [`ChainClient::expire_stalled_training`].
    async fn reassign_coordinator(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
        new_coordinator: WorkerAddress,
    ) -> Result<(), ChainError>;

    /// Expire a Training job whose coordinator has made no progress for
    /// [`STALL_EXPIRY_BLOCKS`]. Mirrors
    /// `ComputePoolTraining.expireStalledTraining`: callable by the
    /// requester or any joined worker; the job moves to Awaiting with the
    /// challenge window starting now, and is then finalized normally.
    /// This is the worker-side exit from a stalled coordinator.
    async fn expire_stalled_training(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
    ) -> Result<(), ChainError>;

    /// Open a challenge against a worker's step commitment. The
    /// caller proves the disputed leaf is actually in the on-chain
    /// epoch root via `merkle_proof`. Bond is held until resolution.
    // The arguments intentionally mirror the challengeStep ABI fields;
    // changing this trait to a wrapper struct would alter the client API.
    #[allow(clippy::too_many_arguments)]
    async fn challenge_step(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
        epoch: EpochIndex,
        step: u32,
        target: WorkerAddress,
        leaf: B256,
        merkle_proof: Vec<B256>,
        bond: u128,
    ) -> Result<(), ChainError>;

    /// Committee member casts a vote on an active challenge. Quorum
    /// of either side triggers automatic resolution.
    async fn vote_challenge(
        &self,
        job_id: JobId,
        voter: WorkerAddress,
        epoch: EpochIndex,
        step: u32,
        target: WorkerAddress,
        uphold: bool,
    ) -> Result<(), ChainError>;

    /// Governance helper: add or remove a committee member. Mock
    /// only — production has a separate governance contract.
    async fn set_committee_member(&self, member: WorkerAddress, active: bool);

    /// Read aggregate slash amount on a (job, worker) tuple — sum
    /// of liveness slash + challenge slash.
    async fn worker_total_slashed(&self, job_id: JobId, worker: WorkerAddress) -> u128;

    /// Read accumulated challenger reward (across all jobs in the
    /// mock; production tracks per-job).
    async fn challenger_reward(&self, challenger: WorkerAddress) -> u128;
}

/// In-memory implementation of the ComputePoolTraining state
/// machine. Mirrors the Solidity contract's invariants byte-for-
/// byte (join → close → commit × E → finalize) so tests
/// exercising the worker state machine catch the same class of
/// bugs they'd catch against a live deployment.
pub struct MockChainClient {
    inner: Arc<Mutex<MockState>>,
    /// See [`MockChainClient::new_live_settlement`]. Defaults to false so every
    /// existing harness is unaffected.
    live_settlement: bool,
}

struct MockState {
    jobs: HashMap<JobId, MockJob>,
    next_job_id: JobId,
    block_number: u64,
    committee: HashSet<WorkerAddress>,
    challenger_rewards: HashMap<WorkerAddress, u128>,
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
    // (epoch, step, target) → challenge record
    challenges: HashMap<(EpochIndex, u32, WorkerAddress), MockChallenge>,
    // Per-worker challenge-slash total.
    challenge_slashed: HashMap<WorkerAddress, u128>,
}

#[derive(Clone)]
struct MockChallenge {
    challenger: WorkerAddress,
    bond: u128,
    state: MockChallengeState,
    uphold_votes: u32,
    reject_votes: u32,
    voted: HashSet<WorkerAddress>,
}

#[derive(Clone, PartialEq, Eq)]
enum MockChallengeState {
    Voting,
    ResolvedUphold,
    ResolvedReject,
}

/// Committee quorum mirroring the contract's COMMITTEE_QUORUM.
const COMMITTEE_QUORUM: u32 = 2;
/// Slash basis points mirroring the contract's SLASH_BPS (10%).
const SLASH_BPS: u128 = 1000;
/// Basis points denominator.
const BPS: u128 = 10_000;

impl MockChainClient {
    pub fn new() -> Arc<Self> {
        Self::with_live_settlement(false)
    }

    /// A mock that reports LIVE SETTLEMENT — it stands in for `HttpChainClient`
    /// against a real deployment, where a committed epoch pays real SALT.
    ///
    /// Exists so the capability gate in `Worker::run` can be tested without a
    /// live chain. The state machine is identical; only the honesty bit differs.
    pub fn new_live_settlement() -> Arc<Self> {
        Self::with_live_settlement(true)
    }

    fn with_live_settlement(live_settlement: bool) -> Arc<Self> {
        Arc::new(Self {
            live_settlement,
            inner: Arc::new(Mutex::new(MockState {
                jobs: HashMap::new(),
                next_job_id: 0,
                block_number: 0,
                committee: HashSet::new(),
                challenger_rewards: HashMap::new(),
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
                challenges: HashMap::new(),
                challenge_slashed: HashMap::new(),
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
            // Fail-safe: a defaulted mock never claims live settlement.
            live_settlement: false,
            inner: Arc::new(Mutex::new(MockState {
                jobs: HashMap::new(),
                next_job_id: 0,
                block_number: 0,
                committee: HashSet::new(),
                challenger_rewards: HashMap::new(),
            })),
        }
    }
}

#[async_trait]
impl ChainClient for MockChainClient {
    fn is_live_settlement(&self) -> bool {
        self.live_settlement
    }

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
        // RM-G2.4 / audit F-4: parity with the on-chain
        // `require(root != bytes32(0), ...)` in
        // ComputePoolTraining.sol::commitEpoch. Pre-fix the mock
        // accepted a zero root and the worker tests passed; the
        // contract would have reverted, so a worker that was OK
        // against the mock could fail in production.
        if root == B256::default() {
            return Err(ChainError::ZeroEpochRoot);
        }

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

    async fn expire_stalled_training(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
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
        if caller != MOCK_REQUESTER && !job.joined.contains(&caller) {
            return Err(ChainError::NotRequesterOrGovernance);
        }
        if block <= job.last_activity_block + STALL_EXPIRY_BLOCKS {
            return Err(ChainError::NotStalled);
        }
        job.state = JobChainState::Awaiting;
        job.all_epochs_committed_block = block;
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
        // Only the requester or governance appoints the coordinator; a
        // joined worker's exit is `expire_stalled_training`.
        if caller != MOCK_REQUESTER && caller != MOCK_GOVERNANCE {
            return Err(ChainError::NotRequesterOrGovernance);
        }
        if !job.joined.contains(&new_coordinator) {
            return Err(ChainError::ReplacementNotJoined);
        }
        if block <= job.last_activity_block + COORDINATION_TIMEOUT {
            return Err(ChainError::CoordinatorStillActive);
        }
        // The liveness slash applies only when governance adjudicates.
        if caller == MOCK_GOVERNANCE {
            if let Some(old) = job.coordinator {
                job.liveness_slashed.insert(old);
            }
        }
        job.coordinator = Some(new_coordinator);
        job.last_activity_block = block;
        Ok(())
    }

    async fn challenge_step(
        &self,
        job_id: JobId,
        caller: WorkerAddress,
        epoch: EpochIndex,
        step: u32,
        target: WorkerAddress,
        leaf: B256,
        merkle_proof: Vec<B256>,
        bond: u128,
    ) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        let job = state
            .jobs
            .get_mut(&job_id)
            .ok_or(ChainError::UnknownJob(job_id))?;
        if caller == target {
            return Err(ChainError::SelfChallenge);
        }
        if !job.joined.contains(&target) {
            return Err(ChainError::TargetNotJoined);
        }
        // RM-G2.4 / audit F-4: mirror the on-chain bond + range
        // checks in ComputePoolTraining.sol::challengeStep. Pre-fix
        // the mock accepted any bond and any step/epoch — a worker
        // that filed a malformed challenge in tests would discover
        // the mismatch only when it hit the live contract.
        if bond != CHALLENGE_BOND {
            return Err(ChainError::WrongChallengeBond);
        }
        if step >= job.spec.steps_per_epoch {
            return Err(ChainError::StepOutOfRange);
        }
        if epoch >= job.spec.epoch_count {
            return Err(ChainError::ChallengeEpochOutOfRange);
        }
        let root = job
            .epoch_roots
            .get(&epoch)
            .copied()
            .ok_or(ChainError::EpochNotCommitted)?;
        // Mirror ComputePoolTraining._verifyMerkleProof — sorted-pair
        // concat + keccak256. Must match the contract bit-for-bit so
        // a proof that verifies here verifies on-chain.
        if !verify_merkle_proof(&merkle_proof, root, leaf) {
            return Err(ChainError::BadMerkleProof);
        }
        let key = (epoch, step, target);
        // Only allow a new challenge if the slot is empty or the
        // previous challenge resolved. Matches the contract's
        // "must be None | Resolved*" gate.
        if let Some(existing) = job.challenges.get(&key) {
            if existing.state == MockChallengeState::Voting {
                return Err(ChainError::ChallengeAlreadyActive);
            }
        }
        job.challenges.insert(
            key,
            MockChallenge {
                challenger: caller,
                bond,
                state: MockChallengeState::Voting,
                uphold_votes: 0,
                reject_votes: 0,
                voted: HashSet::new(),
            },
        );
        Ok(())
    }

    async fn vote_challenge(
        &self,
        job_id: JobId,
        voter: WorkerAddress,
        epoch: EpochIndex,
        step: u32,
        target: WorkerAddress,
        uphold: bool,
    ) -> Result<(), ChainError> {
        let mut state = self.inner.lock();
        if !state.committee.contains(&voter) {
            return Err(ChainError::VoterNotOnCommittee);
        }
        let spec_stake = {
            let job = state
                .jobs
                .get(&job_id)
                .ok_or(ChainError::UnknownJob(job_id))?;
            job.spec.per_worker_stake
        };
        let job = state.jobs.get_mut(&job_id).expect("checked above");
        let key = (epoch, step, target);
        let ch = job
            .challenges
            .get_mut(&key)
            .ok_or(ChainError::ChallengeNotVoting)?;
        if ch.state != MockChallengeState::Voting {
            return Err(ChainError::ChallengeNotVoting);
        }
        if !ch.voted.insert(voter) {
            return Err(ChainError::AlreadyVoted);
        }
        if uphold {
            ch.uphold_votes += 1;
        } else {
            ch.reject_votes += 1;
        }

        // Quorum check (mirroring contract's auto-resolve).
        let decide_uphold = ch.uphold_votes >= COMMITTEE_QUORUM;
        let decide_reject = ch.reject_votes >= COMMITTEE_QUORUM;
        if !decide_uphold && !decide_reject {
            return Ok(());
        }

        let challenger = ch.challenger;
        let bond = ch.bond;

        if decide_uphold {
            ch.state = MockChallengeState::ResolvedUphold;
            ch.bond = 0;
            // Slash target: SLASH_BPS of posted stake, bounded by
            // what remains held. Track cumulative challenge slash.
            let prior_slash = *job.challenge_slashed.get(&target).unwrap_or(&0);
            let liveness_slash_count = if job.liveness_slashed.contains(&target) {
                1
            } else {
                0
            };
            let nominal_slash = spec_stake * SLASH_BPS / BPS;
            // Held = posted - (liveness-slash portion) - (prior challenge slashes).
            // Liveness slash in the Solidity side is also SLASH_BPS? No —
            // the mock models liveness as a boolean; to match contract
            // accounting we treat each liveness-slash event as SLASH_BPS
            // too. Kept conservative: don't over-slash.
            let held_lower_bound = spec_stake
                .saturating_sub(prior_slash)
                .saturating_sub((liveness_slash_count as u128) * (spec_stake * 10 / BPS));
            let slash = nominal_slash.min(held_lower_bound);
            *job.challenge_slashed.entry(target).or_insert(0) += slash;

            // Challenger reward: bond refund + half-slash.
            let reward = bond + slash / 2;
            *state.challenger_rewards.entry(challenger).or_insert(0) += reward;
        } else {
            ch.state = MockChallengeState::ResolvedReject;
            ch.bond = 0;
            // Bond stays with the contract (forfeit).
        }
        Ok(())
    }

    async fn set_committee_member(&self, member: WorkerAddress, active: bool) {
        let mut state = self.inner.lock();
        if active {
            state.committee.insert(member);
        } else {
            state.committee.remove(&member);
        }
    }

    async fn worker_total_slashed(&self, job_id: JobId, worker: WorkerAddress) -> u128 {
        let state = self.inner.lock();
        let Some(job) = state.jobs.get(&job_id) else {
            return 0;
        };
        let challenge_portion = *job.challenge_slashed.get(&worker).unwrap_or(&0);
        let liveness_portion = if job.liveness_slashed.contains(&worker) {
            // Match contract: LIVENESS_SLASH_BPS = 10 (0.1%)
            job.spec.per_worker_stake * 10 / BPS
        } else {
            0
        };
        challenge_portion + liveness_portion
    }

    async fn challenger_reward(&self, challenger: WorkerAddress) -> u128 {
        let state = self.inner.lock();
        *state.challenger_rewards.get(&challenger).unwrap_or(&0)
    }
}

/// Merkle proof verification matching `ComputePoolTraining._verifyMerkleProof`.
///
/// Split out as a free function so tests can call it without holding the
/// MockState lock.
///
/// RM-I / WP-I1.8 (re-audit Stream 3 finding F-4):
///   The contract (`contracts/src/ComputePoolTraining.sol::_verifyMerkleProof`,
///   ~line 764, post-SOL-20) prefixes leaf hashes with `0x00` and internal-
///   node hashes with `0x01` to prevent second-preimage attacks where an
///   internal hash from a larger tree could be presented as a leaf in a
///   smaller tree. The mock's prior implementation did plain sorted-pair
///   concat without prefixes; workers generated proofs that passed the mock
///   tests and would have failed on-chain at admission. RM-I closes the
///   parity gap so the mock and contract produce byte-identical roots.
///
/// Domain separators (must match `contracts/src/ComputePoolTraining.sol:761-762`):
const MERKLE_LEAF_PREFIX: u8 = 0x00;
const MERKLE_INTERNAL_PREFIX: u8 = 0x01;

fn verify_merkle_proof(proof: &[B256], root: B256, leaf: B256) -> bool {
    use sha3::{Digest, Keccak256};
    // Promote the raw leaf hash into the leaf domain (0x00 || leaf).
    let mut computed = {
        let mut h = Keccak256::new();
        h.update([MERKLE_LEAF_PREFIX]);
        h.update(leaf.as_bytes());
        let out = h.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&out);
        B256::from(bytes)
    };
    for sibling in proof {
        let (left, right) = if computed.as_bytes() <= sibling.as_bytes() {
            (computed, *sibling)
        } else {
            (*sibling, computed)
        };
        // Internal node: keccak(0x01 || left || right) — matches the
        // contract's `keccak256(abi.encodePacked(MERKLE_INTERNAL_PREFIX,
        // computed, sibling))` ordering.
        let mut hasher = Keccak256::new();
        hasher.update([MERKLE_INTERNAL_PREFIX]);
        hasher.update(left.as_bytes());
        hasher.update(right.as_bytes());
        let out = hasher.finalize();
        let mut h = [0u8; 32];
        h.copy_from_slice(&out);
        computed = B256::from(h);
    }
    computed == root
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
        let err = chain
            .reassign_coordinator(job_id, MOCK_REQUESTER, w3)
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::CoordinatorStillActive));

        // Past timeout: the requester swaps without a slash.
        chain.advance_blocks(60).await;
        chain
            .reassign_coordinator(job_id, MOCK_REQUESTER, w3)
            .await
            .unwrap();
        let snap = chain.snapshot(job_id).await.unwrap();
        assert_eq!(snap.coordinator, Some(w3), "coordinator swapped");
        assert!(!chain.was_liveness_slashed(job_id, w1));

        // Governance swap after another stall slashes the stalled one.
        chain.advance_blocks(101).await;
        chain
            .reassign_coordinator(job_id, MOCK_GOVERNANCE, w2)
            .await
            .unwrap();
        assert!(chain.was_liveness_slashed(job_id, w3));
    }

    #[tokio::test]
    async fn joined_worker_cannot_reassign_coordinator() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());
        let (w1, w2, w3) = (
            Address::repeat_byte(1),
            Address::repeat_byte(2),
            Address::repeat_byte(3),
        );
        for w in [w1, w2, w3] {
            chain
                .join_training_job(job_id, w, spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain.close_recruitment(job_id, w1).await.unwrap();
        chain.advance_blocks(200).await;
        let err = chain
            .reassign_coordinator(job_id, w2, w2)
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::NotRequesterOrGovernance));
        assert_eq!(chain.snapshot(job_id).await.unwrap().coordinator, Some(w1));
    }

    #[tokio::test]
    async fn expire_stalled_training_after_stall_expiry() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());
        let (w1, w2) = (Address::repeat_byte(1), Address::repeat_byte(2));
        for w in [w1, w2, Address::repeat_byte(3)] {
            chain
                .join_training_job(job_id, w, spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain.close_recruitment(job_id, w1).await.unwrap();

        // Exactly STALL_EXPIRY_BLOCKS since activity: still not stalled.
        chain.advance_blocks(STALL_EXPIRY_BLOCKS).await;
        let err = chain.expire_stalled_training(job_id, w2).await.unwrap_err();
        assert!(matches!(err, ChainError::NotStalled));
        chain.advance_blocks(1).await;

        // An outsider cannot expire it.
        let err = chain
            .expire_stalled_training(job_id, Address::repeat_byte(0xEE))
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::NotRequesterOrGovernance));

        chain.expire_stalled_training(job_id, w2).await.unwrap();
        let snap = chain.snapshot(job_id).await.unwrap();
        assert_eq!(snap.state, JobChainState::Awaiting);
        // Nothing is paid before the challenge window runs from expiry.
        let err = chain.finalize(job_id).await.unwrap_err();
        assert!(matches!(err, ChainError::ChallengeWindowOpen));
        chain
            .advance_blocks(spec().challenge_window_blocks as u64)
            .await;
        chain.finalize(job_id).await.unwrap();
        // Expiry is one-shot.
        let err = chain.expire_stalled_training(job_id, w2).await.unwrap_err();
        assert!(matches!(err, ChainError::WrongState(_)));
    }

    #[tokio::test]
    async fn requester_can_expire_stalled_training() {
        let chain = MockChainClient::new();
        let job_id = chain.create_job(spec());
        for w in [1u8, 2, 3] {
            chain
                .join_training_job(job_id, Address::repeat_byte(w), spec().per_worker_stake)
                .await
                .unwrap();
        }
        chain
            .close_recruitment(job_id, Address::repeat_byte(1))
            .await
            .unwrap();
        chain.advance_blocks(STALL_EXPIRY_BLOCKS + 1).await;
        chain
            .expire_stalled_training(job_id, MOCK_REQUESTER)
            .await
            .unwrap();
        assert_eq!(
            chain.snapshot(job_id).await.unwrap().state,
            JobChainState::Awaiting
        );
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
        assert!(matches!(err, ChainError::NotRequesterOrGovernance));
        // The replacement must still be a joined worker.
        let err = chain
            .reassign_coordinator(job_id, MOCK_REQUESTER, outsider)
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::ReplacementNotJoined));
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

    // ── RM-G2.4 / audit F-4 — invariant parity with the on-chain
    // ComputePoolTraining contract. Each test mirrors one
    // `require(...)` in `ComputePoolTraining.sol` so a worker that
    // passes against the mock won't surprise-fail against the
    // contract.

    /// Mirrors `ComputePoolTraining.sol` line 367:
    /// `require(root != bytes32(0), "ComputePoolTraining: zero root")`.
    #[tokio::test]
    async fn f4_commit_epoch_rejects_zero_root() {
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
            .commit_epoch(job_id, w1, 0, B256::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::ZeroEpochRoot));
    }

    async fn _f4_setup() -> (Arc<MockChainClient>, JobId, Address, Address) {
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
        chain
            .commit_epoch(job_id, w1, 0, B256::repeat_byte(0xEE))
            .await
            .unwrap();
        (chain, job_id, w1, w2)
    }

    /// Mirrors `ComputePoolTraining.sol` line 537:
    /// `require(msg.value == CHALLENGE_BOND, ...)`.
    #[tokio::test]
    async fn f4_challenge_step_rejects_wrong_bond() {
        let (chain, job_id, w1, w2) = _f4_setup().await;
        // Wrong bond.
        let err = chain
            .challenge_step(
                job_id,
                w2,
                0,
                0,
                w1,
                B256::repeat_byte(0xAA),
                vec![],
                CHALLENGE_BOND - 1,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::WrongChallengeBond));
    }

    /// Mirrors `ComputePoolTraining.sol` line 538:
    /// `require(step < job.stepsPerEpoch, ...)`.
    #[tokio::test]
    async fn f4_challenge_step_rejects_step_out_of_range() {
        let (chain, job_id, w1, w2) = _f4_setup().await;
        // spec().steps_per_epoch is 2, so step 2 is out of range.
        let err = chain
            .challenge_step(
                job_id,
                w2,
                0,
                spec().steps_per_epoch,
                w1,
                B256::repeat_byte(0xAA),
                vec![],
                CHALLENGE_BOND,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::StepOutOfRange));
    }

    /// Mirrors `ComputePoolTraining.sol` line 539:
    /// `require(epoch < job.epochCount, ...)`.
    #[tokio::test]
    async fn f4_challenge_step_rejects_epoch_out_of_range() {
        let (chain, job_id, w1, w2) = _f4_setup().await;
        // spec().epoch_count is 2, so epoch 2 is out of range.
        let err = chain
            .challenge_step(
                job_id,
                w2,
                spec().epoch_count,
                0,
                w1,
                B256::repeat_byte(0xAA),
                vec![],
                CHALLENGE_BOND,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ChainError::ChallengeEpochOutOfRange));
    }

    // ────────────────────────────────────────────────────────────────
    // RM-I / WP-I1.8 — Merkle leaf/internal domain separators (F-4 mock parity).
    //
    // Verifies the mock's `verify_merkle_proof` matches the contract's
    // `_verifyMerkleProof` byte-for-byte: leaves are `keccak(0x00 || leaf)`
    // and internal nodes are `keccak(0x01 || L || R)` (with sorted L<=R).
    // Pre-fix the mock did plain sorted-pair concat without prefixes; a
    // proof that verified in the mock would have been rejected on-chain.
    // ────────────────────────────────────────────────────────────────

    fn keccak(input: &[u8]) -> B256 {
        use sha3::{Digest, Keccak256};
        let out = Keccak256::digest(input);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&out);
        B256::from(bytes)
    }

    /// Build a 2-leaf Merkle root using the contract's prefix scheme so a
    /// caller can construct a proof that verifies in BOTH the mock and the
    /// live contract. Returns `(root, leaf_a_proof_for_b)`.
    fn build_two_leaf_root(leaf_a: B256, leaf_b: B256) -> (B256, Vec<B256>) {
        // Domain-promote each leaf.
        let promoted_a = keccak(&[&[0x00u8] as &[u8], leaf_a.as_bytes()].concat());
        let promoted_b = keccak(&[&[0x00u8] as &[u8], leaf_b.as_bytes()].concat());
        // Sorted internal hash with prefix.
        let (l, r) = if promoted_a.as_bytes() <= promoted_b.as_bytes() {
            (promoted_a, promoted_b)
        } else {
            (promoted_b, promoted_a)
        };
        let mut buf = vec![0x01u8];
        buf.extend_from_slice(l.as_bytes());
        buf.extend_from_slice(r.as_bytes());
        let root = keccak(&buf);
        // Proof for leaf_a is just leaf_b (its sibling).
        (root, vec![promoted_b])
    }

    #[test]
    fn test_wp_i1_8_two_leaf_proof_verifies_with_prefixes() {
        let leaf_a = B256::repeat_byte(0xAA);
        let leaf_b = B256::repeat_byte(0xBB);
        let (root, proof) = build_two_leaf_root(leaf_a, leaf_b);
        assert!(
            super::verify_merkle_proof(&proof, root, leaf_a),
            "WP-I1.8: a proof built with the contract's prefix scheme \
             must verify in the mock's verify_merkle_proof."
        );
    }

    #[test]
    fn test_wp_i1_8_unprefixed_proof_does_not_verify() {
        // Pre-fix path: build root WITHOUT prefixes, expect verification to fail
        // because `verify_merkle_proof` now uses prefixes.
        let leaf_a = B256::repeat_byte(0xAA);
        let leaf_b = B256::repeat_byte(0xBB);
        // Plain sorted-pair (no prefixes).
        let (l, r) = if leaf_a.as_bytes() <= leaf_b.as_bytes() {
            (leaf_a, leaf_b)
        } else {
            (leaf_b, leaf_a)
        };
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(l.as_bytes());
        buf.extend_from_slice(r.as_bytes());
        let unprefixed_root = keccak(&buf);
        let proof = vec![leaf_b];
        assert!(
            !super::verify_merkle_proof(&proof, unprefixed_root, leaf_a),
            "WP-I1.8: a proof built WITHOUT the contract's prefix scheme \
             must NOT verify (the mock now mirrors the contract's prefixed \
             verification)."
        );
    }

    #[test]
    fn test_wp_i1_8_leaf_internal_collision_rejected() {
        // Second-preimage protection: an internal hash from a larger tree
        // must not be verifiable as a leaf in a smaller tree.
        let bytes = [0x42u8; 32];
        // Treat `bytes` as if it were a "leaf" — but the prefix means
        // `verify_merkle_proof` would compute keccak(0x00 || bytes), not
        // keccak(bytes). Construct a "root" that's just bytes (as if we
        // accepted it as a 1-element proof). The verifier with prefix
        // scheme produces a different hash and rejects.
        let leaf = B256::from(bytes);
        let attempted_root = leaf; // attacker hopes the leaf hash equals the root
        let proof = vec![]; // 0-element proof
        assert!(
            !super::verify_merkle_proof(&proof, attempted_root, leaf),
            "WP-I1.8: leaf-as-root attack must fail (prefix changes the \
             hash domain)."
        );
    }
}
