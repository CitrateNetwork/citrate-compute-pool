//! Turning a coordinator job into a real training run.
//!
//! This is the last link: the coordinator hands out a `JobSpec` whose payload it
//! deliberately does not interpret, and something has to turn that payload into
//! verified artifacts, real steps on a real GPU, and a commitment a challenger
//! can reproduce. That is this module.
//!
//! ## Everything is refused before anything is trained
//!
//! The order of the checks in [`NatJobRunner::run`] is the design. Each one
//! corresponds to a way an honest worker gets slashed, or a dishonest one gets
//! paid, and every one of them is cheaper to fail than a two-day run is to waste:
//!
//! 1. **The payload must parse into a typed job.** A field the worker guesses at
//!    is a field the challenger will disagree about.
//! 2. **Artifacts must verify.** [`ArtifactStore::resolve`] checks the checkpoint
//!    and the manifest against the hashes the job named, and `JobArtifacts`
//!    cannot be constructed without that — so there is no path here that skipped
//!    verification.
//! 3. **The declared commitment grid must match the artifacts'.** This is the one
//!    that would otherwise be silent: a worker committing on the Q16 grid while
//!    the committee resolves on the legacy f32 scale produces honest work that
//!    fails every challenge. Nothing downstream notices, and the worker loses 10%
//!    of its stake for being correct.
//! 4. **The backend must honour the job spec.** `ModelBackend::honors_job_spec()`
//!    is `false` by default precisely so a placeholder cannot collect for work it
//!    did not do.
//!
//! ## What it will not do
//!
//! It does not fetch artifacts. If the corpus and checkpoint are not already in
//! the local store, the job is declined with the hashes that were wanted. Moving
//! 2.4 GB to volunteer machines is a distribution problem — IPFS, torrent, a
//! signed mirror — and solving it badly inside a training runner would mean a
//! half-built fetcher on the path that settles money.

use std::path::PathBuf;

use crate::backend::ModelBackend;
use crate::coordinator_protocol::JobSpec;
use crate::job_artifacts::{ArtifactStore, CommitmentGrid};
use crate::merkle::compute_epoch_root;
use crate::nat_backend::{NatBackend, TrainingParams};
use crate::types::{B256, StepCommit, WorkerAddress};
use crate::zone_delta::SHARED;
use serde::{Deserialize, Serialize};

/// A training job, as carried in the coordinator payload.
///
/// Typed rather than read field-by-field out of a `serde_json::Value`: an absent
/// field must be a refusal, not a default. A worker that silently trains 1 epoch
/// because `epochs` was missing has produced a result nobody asked for and
/// everybody will pay for.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainingJobPayload {
    /// Discriminator, so a runner cannot be handed a job of another kind and
    /// half-understand it.
    pub task: String,
    pub model_start_hash: B256,
    pub dataset_hash: B256,
    /// **Declared per job**, and checked against the artifacts. See the module
    /// note — this is the field whose disagreement is silent and expensive.
    pub commitment_grid: CommitmentGrid,
    pub epoch: u32,
    /// Steps to run in this job. Bounded by the coordinator, not by the worker.
    pub steps: u32,
    /// Which slice of the corpus this worker reads. Disjoint across workers.
    pub worker_shard: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_windows: usize,
    pub shards_per_step: usize,
    pub seed: u64,
}

pub const TASK_TRAIN: &str = "train";

impl TrainingJobPayload {
    fn params(&self) -> TrainingParams {
        TrainingParams {
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            max_windows: self.max_windows,
            shards_per_step: self.shards_per_step,
            seed: self.seed,
        }
    }
}

/// One step, as the challenger will re-derive it.
#[derive(Clone, Debug, Serialize)]
pub struct StepRecord {
    pub step: u32,
    pub commitment: B256,
    pub post_weights: B256,
}

/// What the worker signs and returns.
///
/// Carries the epoch Merkle root because that is the value that reaches the
/// chain, and the per-step commitments because that is what a challenger needs to
/// prove a specific step wrong. Reporting only the root would make the work
/// unfalsifiable, which is the opposite of the point.
#[derive(Clone, Debug, Serialize)]
pub struct TrainingJobResult {
    pub job: String,
    pub task: &'static str,
    /// `candle-cuda`, `candle-metal` or `candle-cpu` — recorded because the
    /// backend a result came from is part of its provenance once the fleet is
    /// heterogeneous, not an implementation detail.
    pub backend: &'static str,
    pub commitment_grid: &'static str,
    pub epoch: u32,
    pub worker_shard: u32,
    pub steps: Vec<StepRecord>,
    pub epoch_root: B256,
    pub final_weights: B256,
    /// From the VERIFIED manifest, never computed here — a worker scoring its own
    /// data quality is a worker grading its own homework for pay.
    pub data_quality_raw: i64,
    /// Per-zone gradient L2 over the job, so a dead zone (ADR-0012) is visible in
    /// the result rather than discovered in a checkpoint months later.
    pub zone_l2: Vec<(String, f64)>,
    pub seconds: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("payload is not a training job: {0}")]
    BadPayload(String),
    #[error("this runner handles {TASK_TRAIN:?} jobs, not {0:?}")]
    WrongTask(String),
    #[error(
        "artifacts are not staged locally: {0}. This worker does not fetch them; \
         stage the checkpoint and corpus into the artifact store first."
    )]
    ArtifactsMissing(String),
    #[error(
        "job declares the {declared} commitment grid but the artifacts are on {actual}. \
         Refusing: committing on a different grid from the one the committee resolves \
         on produces honest work that fails every challenge."
    )]
    GridMismatch {
        declared: &'static str,
        actual: &'static str,
    },
    #[error(
        "backend does not honour the job spec, so its commitments would not \
         correspond to the work the job describes"
    )]
    BackendDoesNotHonorSpec,
    #[error("training failed: {0}")]
    Training(String),
}

/// Runs `train` jobs against the real NAT backend.
pub struct NatJobRunner {
    store: ArtifactStore,
    scratch: PathBuf,
    worker: WorkerAddress,
}

impl NatJobRunner {
    pub fn new(store_root: impl Into<PathBuf>, scratch: impl Into<PathBuf>, worker: WorkerAddress) -> Self {
        Self {
            store: ArtifactStore::new(store_root),
            scratch: scratch.into(),
            worker,
        }
    }

    /// Execute a job, or refuse it with a reason.
    ///
    /// Refusing is a first-class outcome. The coordinator lets the lease expire
    /// and reassigns the work, which is correct for a machine that cannot do it —
    /// and vastly better than a plausible-looking result nobody can reproduce.
    pub async fn run(&self, job: &JobSpec) -> Result<TrainingJobResult, RunError> {
        let payload: TrainingJobPayload = serde_json::from_value(job.payload.clone())
            .map_err(|e| RunError::BadPayload(e.to_string()))?;
        if payload.task != TASK_TRAIN {
            return Err(RunError::WrongTask(payload.task));
        }

        // Verification happens here, not as a courtesy: `JobArtifacts` cannot be
        // constructed without both content hashes matching.
        let artifacts = self
            .store
            .resolve(&payload.model_start_hash, &payload.dataset_hash)
            .map_err(|e| RunError::ArtifactsMissing(e.to_string()))?;

        if artifacts.commitment_grid != payload.commitment_grid {
            return Err(RunError::GridMismatch {
                declared: payload.commitment_grid.as_str(),
                actual: artifacts.commitment_grid.as_str(),
            });
        }

        let backend = NatBackend::new(artifacts, payload.params(), self.scratch.clone())
            .map_err(|e| RunError::Training(e.to_string()))?;

        if !backend.honors_job_spec() {
            return Err(RunError::BackendDoesNotHonorSpec);
        }

        let started = std::time::Instant::now();
        let mut prev = backend
            .load_starting_weights(payload.model_start_hash)
            .await
            .map_err(|e| RunError::Training(e.to_string()))?;

        let mut steps = Vec::with_capacity(payload.steps as usize);
        let mut commits = Vec::with_capacity(payload.steps as usize);
        let mut zone_l2: Vec<(String, f64)> = Vec::new();

        for step in 0..payload.steps {
            let result = backend
                .forward_backward(prev, payload.epoch, step, payload.worker_shard)
                .await
                .map_err(|e| RunError::Training(e.to_string()))?;

            let commitment = backend.compute_step_commitment(&result.gradients);
            accumulate_zone_l2(&mut zone_l2, &result.gradients);

            commits.push(StepCommit {
                epoch: payload.epoch,
                step,
                worker: self.worker,
                commitment,
                prev_weights: prev,
            });
            steps.push(StepRecord {
                step,
                commitment,
                post_weights: result.post_weights_hash,
            });
            // Chained: step s's post-weights are step s+1's prev-weights, which is
            // what makes a mid-run substitution detectable.
            prev = result.post_weights_hash;
        }

        let (epoch_root, _leaves) = compute_epoch_root(&commits);
        let data_quality = backend
            .data_quality()
            .map_err(|e| RunError::Training(e.to_string()))?;

        Ok(TrainingJobResult {
            job: job.id.0.clone(),
            task: TASK_TRAIN,
            backend: backend.backend_tag(),
            commitment_grid: payload.commitment_grid.as_str(),
            epoch: payload.epoch,
            worker_shard: payload.worker_shard,
            steps,
            epoch_root,
            final_weights: prev,
            data_quality_raw: data_quality.raw(),
            zone_l2,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

/// Sum per-bucket gradient L2 across steps.
///
/// The backend assigns `layer_index` in sorted zone order, so index 0 is the
/// first zone alphabetically and the shared embedding/readout bucket sorts under
/// [`SHARED`]. Labelled generically here rather than hardcoding NAT's five zone
/// names, which would silently mislabel a model with a different zone set.
fn accumulate_zone_l2(acc: &mut Vec<(String, f64)>, gradients: &[crate::backend::Tensor]) {
    for t in gradients {
        let l2 = t.data.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        let label = if t.layer_index == usize::MAX {
            SHARED.to_string()
        } else {
            format!("bucket_{}", t.layer_index)
        };
        match acc.iter_mut().find(|(n, _)| *n == label) {
            Some((_, v)) => *v += l2,
            None => acc.push((label, l2)),
        }
    }
}

#[cfg(test)]
mod tests {
    include!("job_runner_tests.rs");
}
