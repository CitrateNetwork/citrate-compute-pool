//! Worker state machine.
//!
//! A single worker runs through the full DataParallelTrainingJob
//! lifecycle from the planset:
//!
//!   Joining → Loading → Syncing → Training (per-epoch: per-step
//!   commit + maybe aggregate) → Awaiting → Settled
//!
//! Any worker CAN be elected coordinator for an epoch; the role
//! is handled inside the same state machine — when `is_coordinator`
//! for the current epoch, the worker also runs the aggregation
//! path (collect step commits from peers, compute Merkle root,
//! post commitEpoch tx).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::backend::ModelBackend;
use crate::chain::{ChainClient, JobChainState};
use crate::merkle::compute_epoch_root;
use crate::transport::{Transport, WorkerMessage};
use crate::types::{B256, EpochIndex, JobId, StepCommit, StepIndex, WorkerAddress};

/// SECREM-02 6.3 (CITRATE_COMPUTE_POOL-2026-05-31-005): per-message
/// timeout on the epoch drain loops. A peer that stops sending (or a
/// flooder that never sends a counted commit) can no longer wedge the
/// worker on `recv` forever — the epoch fails loudly instead.
const DRAIN_RECV_TIMEOUT: Duration = Duration::from_secs(60);

/// SECREM-02 6.3 (-005): overall deadline for the NON-coordinator
/// archive drain. The archive is best-effort (challenge-time lookup
/// cache); a stalled mesh must not block epoch progression, so on
/// deadline we log + move on to waiting for the on-chain root.
const ARCHIVE_DRAIN_DEADLINE: Duration = Duration::from_secs(120);

/// CP-B-010: upper bound on how long a non-coordinator waits for the
/// coordinator's epoch root to appear in the chain snapshot before
/// bailing, so a never-populated `epoch_roots` cannot spin forever.
const EPOCH_ROOT_WAIT_DEADLINE: Duration = Duration::from_secs(600);

/// RM-E.3 / COMPUTE_POOL-001 — authenticated epoch-commit aggregation.
///
/// The coordinator's Merkle root drives reward/slash distribution, so the
/// leaf set must be trustworthy. Pre-fix the aggregation counted any
/// `StepCommit` whose `epoch` matched, with no membership check and no
/// dedup — a single peer could flood `expected_total` forged commits
/// (arbitrary `worker`/`step`/`commitment`) and finalize a root over an
/// attacker-chosen leaf set before honest commits arrived.
///
/// This aggregator enforces, transport-agnostically:
///   - **membership**: `commit.worker` must be in the on-chain worker set,
///   - **dedup**: at most one accepted commit per `(worker, step)` pair.
///
/// (Sender authenticity — `verified_sender == commit.worker` — is enforced
/// at the libp2p transport boundary where `verify_envelope` yields the
/// signer; the InProcess test transport is trusted.)
pub(crate) struct EpochAggregator {
    epoch: EpochIndex,
    members: HashSet<WorkerAddress>,
    seen: HashSet<(WorkerAddress, StepIndex)>,
    accepted: Vec<StepCommit>,
}

impl EpochAggregator {
    pub(crate) fn new(
        epoch: EpochIndex,
        members: impl IntoIterator<Item = WorkerAddress>,
    ) -> Self {
        Self {
            epoch,
            members: members.into_iter().collect(),
            seen: HashSet::new(),
            accepted: Vec::new(),
        }
    }

    /// Try to accept a commit into the aggregation. Returns `true` only if
    /// it is for this epoch, from a registered worker, and the first commit
    /// seen for its `(worker, step)` pair. Forged (non-member) and duplicate
    /// commits are rejected so they cannot reach `expected_total` or alter
    /// the root.
    pub(crate) fn try_accept(&mut self, commit: StepCommit) -> bool {
        if commit.epoch != self.epoch {
            return false;
        }
        if !self.members.contains(&commit.worker) {
            return false;
        }
        if !self.seen.insert((commit.worker, commit.step)) {
            return false;
        }
        self.accepted.push(commit);
        true
    }

    pub(crate) fn len(&self) -> usize {
        self.accepted.len()
    }

    pub(crate) fn into_commits(self) -> Vec<StepCommit> {
        self.accepted
    }
}

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    pub job_id: JobId,
    pub self_address: WorkerAddress,
    pub shard_index: u32,
    /// If this worker is the coordinator for epoch N, their address
    /// matches the elected coordinator in the chain snapshot at the
    /// start of epoch N. For S0 we keep a single coordinator for
    /// the whole job (set at close_recruitment); VRF rotation per
    /// epoch is a future slice mirroring CM-05.
    pub is_coordinator: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerOutcome {
    /// Full lifecycle completed — every epoch committed, finalize
    /// succeeded.
    Finalized,
    /// Job was aborted during recruiting.
    Aborted,
    /// Worker terminated early (e.g., drop-out scenario).
    Terminated(String),
}

pub struct Worker<M, T, C>
where
    M: ModelBackend + 'static,
    T: Transport + 'static,
    C: ChainClient + 'static,
{
    pub config: WorkerConfig,
    pub backend: Arc<M>,
    pub transport: Arc<T>,
    pub chain: Arc<C>,
}

impl<M, T, C> Worker<M, T, C>
where
    M: ModelBackend + 'static,
    T: Transport + 'static,
    C: ChainClient + 'static,
{
    pub fn new(config: WorkerConfig, backend: Arc<M>, transport: Arc<T>, chain: Arc<C>) -> Self {
        Self {
            config,
            backend,
            transport,
            chain,
        }
    }

    /// Run the full lifecycle in this tokio task. Returns when the
    /// job reaches a terminal chain state. Designed to be driven
    /// via `tokio::spawn`.
    pub async fn run(self) -> anyhow::Result<WorkerOutcome> {
        let job_id = self.config.job_id;
        let me = self.config.self_address;

        info!(
            worker = %me,
            job_id = job_id,
            is_coord = self.config.is_coordinator,
            "worker starting"
        );

        // CAPABILITY GATE — refuse to earn real money for placeholder work.
        //
        // A worker is paid per epoch for committing a Merkle root of its step
        // commitments, and the protocol cannot distinguish a root produced by
        // real training from one produced by a harness — both are just hashes.
        // So the honesty check has to happen here, before the first commit.
        //
        // Fails CLOSED by construction: both capabilities default to the safe
        // answer, so a backend or chain client written later is refused until
        // someone deliberately asserts it is real.
        if self.chain.is_live_settlement() && !self.backend.honors_job_spec() {
            return Err(anyhow::anyhow!(
                "refusing to run job {job_id}: this chain settles in real SALT but the \
                 model backend reports honors_job_spec() == false, i.e. it does not load \
                 the weights named by model_start_hash or train on the data named by \
                 dataset_hash. Committing epochs from it would collect real payment for \
                 work that was not done, and would be indistinguishable on-chain from an \
                 honest worker. Supply a backend that honours the job spec, or point this \
                 daemon at a non-settling chain."
            ));
        }

        // Load starting weights. For S0 this is a no-op that just
        // returns the hash; S2 backends fetch from IPFS.
        let snap = self.chain.snapshot(job_id).await?;

        // CP-B-010: refuse to run against a live-settlement chain whose
        // snapshot reports zero workers. The HTTP chain client's
        // `snapshot()` currently hard-codes an empty worker set + empty
        // epoch-root map (the S1.5 TODO). Composing it with a live
        // `is_live_settlement()` would (a) commit an all-zero epoch
        // Merkle root via `commitEpoch` — discarding every worker's real
        // commitment and making `challengeStep`'s inclusion proof
        // structurally impossible — (b) underflow `worker_count - 1` on
        // the non-coordinator path, and (c) spin an unbounded epoch-root
        // poll loop because `epoch_roots` is always empty. Fail closed.
        if self.chain.is_live_settlement() && snap.workers.is_empty() {
            return Err(anyhow::anyhow!(
                "refusing to run job {job_id}: the chain reports live settlement but the \
                 chain snapshot contains zero workers. Committing epochs from this snapshot \
                 would post an all-zero Merkle root and discard every real commitment, \
                 leaving honest work indistinguishable from no work at settlement time. \
                 This is the partial-snapshot trap: populate workers/epoch_roots (or return \
                 is_live_settlement() == false) before settling."
            ));
        }

        let mut prev_weights = self
            .backend
            .load_starting_weights(snap.spec.model_start_hash)
            .await?;
        debug!(worker = %me, "loaded starting weights");

        let spec = snap.spec.clone();
        let worker_count = snap.workers.len() as u32;

        // Per-epoch loop.
        for epoch in 0..spec.epoch_count {
            debug!(worker = %me, epoch = epoch, "entering epoch");

            // SECREM-02 6.3 (-002): tell the transport the current
            // epoch so outgoing envelopes are bound to it and
            // inbound envelopes outside the window are rejected.
            self.transport.set_epoch(epoch).await;

            // Coordinator collects step commits on a side-channel
            // as workers broadcast them. Non-coordinator workers
            // also archive peer commits (for challenge-time
            // lookup), but don't aggregate.
            let mut my_commits_this_epoch: Vec<StepCommit> = Vec::new();
            let mut all_peer_commits: HashMap<u32, Vec<StepCommit>> = HashMap::new();

            for step in 0..spec.steps_per_epoch {
                // Compute our step locally.
                let result = self
                    .backend
                    .forward_backward(prev_weights, epoch, step, self.config.shard_index)
                    .await?;
                let commitment = self.backend.compute_step_commitment(&result.gradients);
                let step_commit = StepCommit {
                    epoch,
                    step,
                    worker: me,
                    commitment,
                    prev_weights,
                };

                // Advance our local weight chain.
                prev_weights = result.post_weights_hash;

                my_commits_this_epoch.push(step_commit.clone());

                // Broadcast to peers.
                self.transport
                    .broadcast(WorkerMessage::StepCommitted(step_commit.clone()))
                    .await?;

                debug!(
                    worker = %me,
                    epoch = epoch,
                    step = step,
                    "emitted step commit"
                );
            }

            // If we're the coordinator, drain the inbox until we've
            // seen every other worker's (steps-per-epoch) commits for
            // this epoch, then compute + post the Merkle root.
            if self.config.is_coordinator {
                let expected_total =
                    (worker_count as usize) * (spec.steps_per_epoch as usize);

                // RM-E.3 / COMPUTE_POOL-001: aggregate through a
                // membership + (worker, step) dedup gate so a flooder
                // cannot stuff forged or duplicate leaves into the root,
                // nor prematurely satisfy `expected_total`. The verified
                // gossip sender is bound to `commit.worker` at the libp2p
                // transport boundary (see libp2p_transport::verify path);
                // here we additionally require on-chain worker-set
                // membership and uniqueness.
                let mut agg = EpochAggregator::new(epoch, snap.workers.iter().copied());
                for c in &my_commits_this_epoch {
                    agg.try_accept(c.clone());
                }

                while agg.len() < expected_total {
                    // SECREM-02 6.3 (-005): per-message timeout so a
                    // silent mesh can't wedge the coordinator forever.
                    let recv = tokio::time::timeout(DRAIN_RECV_TIMEOUT, self.transport.recv(me))
                        .await
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "timed out waiting for step commits (epoch {epoch}: {}/{expected_total} collected)",
                                agg.len()
                            )
                        })?;
                    if let Some(msg) = recv {
                        if let WorkerMessage::StepCommitted(commit) = msg {
                            if agg.try_accept(commit.clone()) {
                                all_peer_commits
                                    .entry(commit.step)
                                    .or_default()
                                    .push(commit);
                            } else {
                                warn!(
                                    worker = %me,
                                    epoch = epoch,
                                    peer = %commit.worker,
                                    "rejected non-member / duplicate / wrong-epoch step commit"
                                );
                            }
                        }
                    } else {
                        anyhow::bail!("transport closed during aggregation");
                    }
                }

                let (root, _leaves) = compute_epoch_root(&agg.into_commits());
                debug!(
                    worker = %me,
                    epoch = epoch,
                    "coordinator computed root"
                );

                self.chain.commit_epoch(job_id, me, epoch, root).await?;
                info!(worker = %me, epoch = epoch, "coordinator posted epoch root");
            } else {
                // Non-coordinator: drain own inbox for peer commits
                // so the archive is populated. We expect
                // (worker_count - 1) × steps_per_epoch UNIQUE commits.
                //
                // SECREM-02 6.3 (-005): the drain is gated through the
                // same membership + (worker, step) dedup as the
                // coordinator aggregation (reusing EpochAggregator),
                // counts only unique accepted commits, and is bounded
                // by a per-message timeout + an overall deadline so a
                // flooder or silent mesh can't wedge the worker.
                // CP-B-010: `saturating_sub` so a zero/one worker_count
                // can never underflow `usize` (panic in debug, wrap to
                // usize::MAX in release) on this non-coordinator path.
                let expected =
                    (worker_count as usize).saturating_sub(1) * (spec.steps_per_epoch as usize);
                let mut archive_gate =
                    EpochAggregator::new(epoch, snap.workers.iter().copied());
                let deadline = tokio::time::Instant::now() + ARCHIVE_DRAIN_DEADLINE;
                while archive_gate.len() < expected {
                    let per_msg_deadline = std::cmp::min(
                        deadline,
                        tokio::time::Instant::now() + DRAIN_RECV_TIMEOUT,
                    );
                    let recv =
                        tokio::time::timeout_at(per_msg_deadline, self.transport.recv(me)).await;
                    let msg = match recv {
                        Ok(Some(msg)) => msg,
                        Ok(None) => {
                            warn!(worker = %me, epoch = epoch, "transport closed during archive drain");
                            break;
                        }
                        Err(_) => {
                            warn!(
                                worker = %me,
                                epoch = epoch,
                                archived = archive_gate.len(),
                                expected = expected,
                                "archive drain deadline reached; proceeding with partial archive"
                            );
                            break;
                        }
                    };
                    if let WorkerMessage::StepCommitted(commit) = msg {
                        if commit.worker != me && archive_gate.try_accept(commit.clone()) {
                            all_peer_commits
                                .entry(commit.step)
                                .or_default()
                                .push(commit);
                        }
                    }
                }
                // Wait for the coordinator to post the epoch root on
                // chain — poll the chain snapshot. On S0 the mock
                // responds immediately so this loop exits in O(1).
                //
                // CP-B-010: bound the wait with a deadline so a chain
                // client whose snapshot never populates `epoch_roots`
                // (the S1.5 partial-snapshot trap) cannot spin this loop
                // forever issuing eth_calls. On a live chain the epoch
                // root should appear within the coordination timeout.
                let root_deadline = tokio::time::Instant::now() + EPOCH_ROOT_WAIT_DEADLINE;
                loop {
                    let snap = self.chain.snapshot(job_id).await?;
                    if snap.epoch_roots.contains_key(&epoch) {
                        break;
                    }
                    if tokio::time::Instant::now() >= root_deadline {
                        anyhow::bail!(
                            "epoch {epoch}: coordinator epoch root did not appear within \
                             the wait deadline (chain snapshot never populated epoch_roots)"
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        }

        // Awaiting → wait for challenge window to pass, then anyone
        // (let's say coordinator) calls finalize.
        if self.config.is_coordinator {
            // Roll blocks forward to simulate the challenge window
            // elapsing. Real worker waits for the real block clock.
            self.chain.advance_blocks(spec.challenge_window_blocks as u64 + 1).await;
            self.chain.finalize(job_id).await?;
            info!(worker = %me, "finalized job");
        } else {
            // Wait for finalize.
            loop {
                let snap = self.chain.snapshot(job_id).await?;
                match snap.state {
                    JobChainState::Finalized => break,
                    JobChainState::Aborted => return Ok(WorkerOutcome::Aborted),
                    _ => tokio::time::sleep(std::time::Duration::from_millis(5)).await,
                }
            }
        }

        info!(worker = %me, "reached terminal state Finalized");
        Ok(WorkerOutcome::Finalized)
    }
}

/// Spawn a worker onto the tokio runtime.
pub fn spawn_worker<M, T, C>(worker: Worker<M, T, C>) -> JoinHandle<anyhow::Result<WorkerOutcome>>
where
    M: ModelBackend + 'static,
    T: Transport + 'static,
    C: ChainClient + 'static,
{
    tokio::spawn(worker.run())
}

#[allow(dead_code)]
fn _type_check() {
    // Guarantee Send bounds — if any field isn't Send, this will
    // fail to compile.
    fn assert_send<T: Send>() {}
    assert_send::<B256>();
}

#[cfg(test)]
mod compute_pool_001_tests {
    use super::*;
    use crate::merkle::compute_epoch_root;
    use ethereum_types::{Address, H256};

    fn addr(b: u8) -> WorkerAddress {
        Address::from([b; 20])
    }
    fn commit(worker: WorkerAddress, epoch: u32, step: u32, c: u8) -> StepCommit {
        StepCommit {
            epoch,
            step,
            worker,
            commitment: H256::from([c; 32]),
            prev_weights: H256::zero(),
        }
    }

    /// RM-E.3 / COMPUTE_POOL-001 tripwire: forged (non-member) and
    /// duplicate `(worker, step)` commits must NOT be counted into the
    /// aggregation or alter the epoch Merkle root. Pre-fix the loop pushed
    /// every epoch-matching commit unconditionally, so this stream would
    /// have polluted the root and could have short-circuited `expected_total`.
    #[test]
    fn tripwire_001_aggregator_rejects_forged_and_duplicate_commits() {
        let w1 = addr(1);
        let w2 = addr(2);
        let evil = addr(0xEE); // NOT in the registered worker set
        let members = [w1, w2];
        let epoch = 0u32;

        // Honest baseline: w1 and w2 each commit step 0.
        let honest = vec![commit(w1, epoch, 0, 0x11), commit(w2, epoch, 0, 0x22)];
        let (honest_root, _) = compute_epoch_root(&honest);

        // Hostile stream interleaved with the honest commits.
        let mut agg = EpochAggregator::new(epoch, members);
        assert!(agg.try_accept(commit(w1, epoch, 0, 0x11)), "honest w1 accepted");
        assert!(
            !agg.try_accept(commit(evil, epoch, 0, 0xEE)),
            "forged non-member commit must be rejected"
        );
        assert!(
            !agg.try_accept(commit(w1, epoch, 0, 0x99)),
            "duplicate (w1, step0) with divergent commitment must be rejected"
        );
        assert!(
            !agg.try_accept(commit(w2, epoch + 1, 0, 0x22)),
            "wrong-epoch commit must be rejected"
        );
        assert!(agg.try_accept(commit(w2, epoch, 0, 0x22)), "honest w2 accepted");

        assert_eq!(agg.len(), 2, "only the two honest commits are counted");
        let (got_root, _) = compute_epoch_root(&agg.into_commits());
        assert_eq!(
            got_root, honest_root,
            "forged/duplicate leaves must not change the epoch root"
        );
    }

    /// A flooder emitting only forged/non-member commits can never reach
    /// `expected_total` — the guard removes the premature-finalize vector.
    #[test]
    fn tripwire_001_flood_of_forged_commits_never_counts() {
        let w1 = addr(1);
        let evil = addr(0xEE);
        let mut agg = EpochAggregator::new(0, [w1]);
        for step in 0..100u32 {
            assert!(!agg.try_accept(commit(evil, 0, step, 0xEE)));
        }
        assert_eq!(agg.len(), 0, "no forged commit may be counted");
    }
}
