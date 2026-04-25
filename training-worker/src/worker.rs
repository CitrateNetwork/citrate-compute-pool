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

use std::collections::HashMap;
use std::sync::Arc;

use tokio::task::JoinHandle;
use tracing::{debug, info};

use crate::backend::ModelBackend;
use crate::chain::{ChainClient, JobChainState};
use crate::merkle::compute_epoch_root;
use crate::transport::{Transport, WorkerMessage};
use crate::types::{B256, JobId, StepCommit, WorkerAddress};

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

        // Load starting weights. For S0 this is a no-op that just
        // returns the hash; S2 backends fetch from IPFS.
        let snap = self.chain.snapshot(job_id).await?;
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
                let mut all_commits: Vec<StepCommit> = my_commits_this_epoch.clone();

                while all_commits.len() < expected_total {
                    if let Some(msg) = self.transport.recv(me).await {
                        if let WorkerMessage::StepCommitted(commit) = msg {
                            if commit.epoch == epoch {
                                all_peer_commits
                                    .entry(commit.step)
                                    .or_default()
                                    .push(commit.clone());
                                all_commits.push(commit);
                            }
                        }
                    } else {
                        anyhow::bail!("transport closed during aggregation");
                    }
                }

                let (root, _leaves) = compute_epoch_root(&all_commits);
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
                // (worker_count - 1) × steps_per_epoch messages.
                let expected =
                    ((worker_count as usize) - 1) * (spec.steps_per_epoch as usize);
                for _ in 0..expected {
                    if let Some(msg) = self.transport.recv(me).await {
                        if let WorkerMessage::StepCommitted(commit) = msg {
                            if commit.epoch == epoch {
                                all_peer_commits
                                    .entry(commit.step)
                                    .or_default()
                                    .push(commit);
                            }
                        }
                    }
                }
                // Wait for the coordinator to post the epoch root on
                // chain — poll the chain snapshot. On S0 the mock
                // responds immediately so this loop exits in O(1).
                loop {
                    let snap = self.chain.snapshot(job_id).await?;
                    if snap.epoch_roots.contains_key(&epoch) {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
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
