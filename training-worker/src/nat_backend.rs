//! The real training backend: NAT's autoregressive LMs behind [`ModelBackend`].
//!
//! This is the crate's first backend that may honestly answer
//! `honors_job_spec() == true`, and it earns that from three layers that already
//! landed and are deliberately NOT feature-gated, so the honesty checks compile
//! and test on any machine:
//!
//! - [`crate::job_artifacts`] — the weights and the corpus are the ones the chain
//!   committed to, verified by content hash before a step runs.
//! - [`crate::q16_commitment`] — commitments ride the fixed Q16 grid from
//!   `citrate_fed_types`, the same one NAT and the chain use, so two workers on
//!   different hardware commit identically.
//! - [`crate::zone_delta`] — the step's weight delta is attributed per zone, the
//!   unit the co-op's federated seam aggregates.
//!
//! ## Why a weight delta rather than gradients
//!
//! `forward_backward` is named for gradients, and `nat_candle::AutoregLm` exposes
//! none: `varmap` is private and `train_minibatched` runs its own AdamW loop.
//! Rather than fork NAT to reach inside, this reports the delta the optimizer
//! actually produced — `after - before` — which is what the federated seam wants
//! anyway (`nat_federated::ZoneWeightDelta`).
//!
//! That is a design choice, not a workaround dressed up as one. A raw gradient
//! and an optimizer-applied delta differ (momentum, weight decay, LR schedule),
//! and the delta is the honest description of what this worker contributed to the
//! shared model.
//!
//! The delta is read via `named_parameters()` (NAT ADR-0011) — the parameters
//! straight out of the model. An earlier revision had to `save` to safetensors,
//! read the file back and diff it, once per step; that round-trip is gone.
//!
//! ## Held directly, not on a worker thread
//!
//! An earlier revision confined the model to a dedicated thread behind a command
//! channel, because `AutoregLm` was not `Send`: it holds
//! `Vec<Box<dyn CausalCore>>` and the trait object carried no bound, even though
//! every field it owns already is. NAT ADR-0011 added `CausalCore: Send + Sync`,
//! so the model is held directly under a `Mutex` and the channel is gone.
//!
//! ## Both architectures, neither a fallback for the other
//!
//! [`Architecture::ZonePartitioned`] runs `AutoregLm` (causal attention HP/PF/CX,
//! causal SSM SM/CB; `MX` is the non-learned harness and is never trained).
//! [`Architecture::Dense`] runs `AutoregDenseLm`. Which one is decided by the
//! verified sidecar — and a job asking for anything else, mixture-of-experts
//! included, never reaches this module: `ArtifactStore::resolve` refuses it.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use parking_lot::Mutex;

use nat_candle::autoreg::{AutoregConfig, AutoregDenseLm, AutoregLm};
use nat_data::manifest::{CorpusManifest, Shard, ShardManifest};
use nat_types::ZoneId;

use crate::backend::{ModelBackend, StepResult, Tensor};
use crate::job_artifacts::{Architecture, CommitmentGrid, JobArtifacts};
use crate::q16_commitment::{q16_step_commitment, to_q16};
use crate::quantize::{quantize_tensor, tensor_commitment};
use crate::types::{CommitmentHash, EpochIndex, PrevWeightsHash, StepIndex, WeightsHash};
use crate::zone_delta::zone_deltas;

/// Hyper-parameters for a step. Not read from chain: the job spec carries the
/// artifact hashes and `steps_per_epoch`, not the optimizer setup.
#[derive(Debug, Clone)]
pub struct TrainingParams {
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Windows drawn from the corpus. Bounds memory on a large corpus —
    /// corpus-v6 is 306.5M tokens across 185,475 shards, so this is what keeps a
    /// step from trying to materialise the whole thing.
    pub max_windows: usize,
    /// Shards to read per step. Only enough to fill `max_windows` is read; the
    /// corpus has 185,475 shard FILES and an 80 MB manifest, so slurping it is
    /// not an option.
    pub shards_per_step: usize,
    pub seed: u64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            batch_size: 16,
            learning_rate: 1e-3,
            max_windows: 4096,
            shards_per_step: 64,
            seed: 2026,
        }
    }
}

/// The model, in whichever arm the job asked for.
enum Model {
    Zone(Box<AutoregLm>),
    Dense(Box<AutoregDenseLm>),
}

impl Model {
    /// Parameters as `(name, values)`, name-ordered — NAT ADR-0011. The names
    /// carry zone identity (`zone_HP.wq`, `score_PF`), which is what
    /// `zone_delta` attributes against.
    fn named_parameters(&self) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
        Ok(match self {
            Model::Zone(m) => m.named_parameters()?,
            Model::Dense(m) => m.named_parameters()?,
        })
    }

    fn load(&mut self, dir: &Path) -> anyhow::Result<()> {
        match self {
            Model::Zone(m) => m.load(dir)?,
            Model::Dense(m) => m.load(dir)?,
        }
        Ok(())
    }

    fn device(&self) -> &candle_core::Device {
        match self {
            Model::Zone(m) => m.device(),
            Model::Dense(m) => m.device(),
        }
    }

    /// Mirrors NAT's `trace.backend`, which is how a run proves it was not on
    /// toy cores.
    fn backend_tag(&self) -> &'static str {
        match self.device() {
            candle_core::Device::Cpu => "candle-cpu",
            _ => "candle-cuda",
        }
    }

    fn train_one_pass(
        &mut self,
        ids: &candle_core::Tensor,
        p: &TrainingParams,
        shuffle_seed: u64,
    ) -> anyhow::Result<()> {
        // `epochs = 1` is one OPTIMIZER pass over the drawn windows, not a chain
        // epoch. A chain epoch is a settlement period and contains many of these.
        match self {
            Model::Zone(m) => {
                m.train_minibatched(ids, 1, p.batch_size, p.learning_rate, shuffle_seed)?
            }
            Model::Dense(m) => {
                m.train_minibatched(ids, 1, p.batch_size, p.learning_rate, shuffle_seed)?
            }
        }
        Ok(())
    }
}

fn dtype_of(shape: &crate::job_artifacts::ModelShape) -> anyhow::Result<candle_core::DType> {
    match shape.dtype.as_str() {
        "f32" => Ok(candle_core::DType::F32),
        "bf16" => Ok(candle_core::DType::BF16),
        other => anyhow::bail!(
            "unsupported checkpoint dtype '{other}'. Refusing rather than widening \
             to f32 — the weights would load with different numerics than they \
             were trained with."
        ),
    }
}

fn build_model(
    arch: Architecture,
    shape: &crate::job_artifacts::ModelShape,
    p: &TrainingParams,
) -> anyhow::Result<Model> {
    let dtype = dtype_of(shape)?;
    Ok(match arch {
        Architecture::ZonePartitioned => {
            let cfg = AutoregConfig {
                // The five LEARNED zones. `MX` is the non-learned executive
                // harness: `AutoregLm` bails on a zone with no core, and the
                // federated seam rejects an MX delta — so it is excluded here
                // rather than filtered downstream.
                zones: vec![ZoneId::SM, ZoneId::CB, ZoneId::HP, ZoneId::PF, ZoneId::CX],
                vocab: shape.vocab,
                seq_len: shape.seq_len,
                d: shape.d,
                tau: 1.0,
                seed: p.seed,
            };
            Model::Zone(Box::new(AutoregLm::new_with_dtype(&cfg, dtype)?))
        }
        Architecture::Dense => Model::Dense(Box::new(AutoregDenseLm::new_with_dtype(
            shape.vocab,
            shape.seq_len,
            shape.d,
            shape.d_ff,
            p.seed,
            dtype,
        )?)),
    })
}

/// NAT-backed [`ModelBackend`].
pub struct NatBackend {
    artifacts: JobArtifacts,
    params: TrainingParams,
    model: Mutex<Model>,
}

impl NatBackend {
    /// Build a backend for artifacts that are ALREADY verified.
    ///
    /// Taking [`JobArtifacts`] by value rather than raw hashes is the point: that
    /// type cannot be constructed without both content hashes matching, so there
    /// is no path in here that skipped verification.
    pub fn new(
        artifacts: JobArtifacts,
        params: TrainingParams,
        _scratch: impl Into<PathBuf>,
    ) -> anyhow::Result<Self> {
        let model = build_model(artifacts.architecture, &artifacts.shape, &params)?;
        Ok(Self {
            artifacts,
            params,
            model: Mutex::new(model),
        })
    }

    /// `"candle-cpu"` or `"candle-cuda"`.
    pub fn backend_tag(&self) -> &'static str {
        self.model.lock().backend_tag()
    }

    /// The verified corpus manifest.
    pub fn load_manifest(&self) -> anyhow::Result<CorpusManifest> {
        let raw = std::fs::read_to_string(&self.artifacts.manifest_path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// The corpus `data_quality` for `nat_train::StepContribution`.
    ///
    /// Read from the VERIFIED manifest's `aggregate_quality`, not computed here.
    /// The pipeline's QUALITY_SCORE stage owns that number, and a worker scoring
    /// its own data quality is a worker grading its own homework for pay.
    pub fn data_quality(&self) -> anyhow::Result<citrate_fed_types::Q16> {
        Ok(self.load_manifest()?.aggregate_quality)
    }

    /// Read a BOUNDED set of shards and verify each against the manifest's
    /// committed `provenance_root`.
    ///
    /// `CorpusManifest` is METADATA: its `shards` are per-shard counts, quality
    /// and a provenance root — not the documents. So `dataset_hash` verifying the
    /// manifest is necessary and NOT sufficient. Recomputing each shard's root is
    /// what actually binds the bytes trained on to the on-chain hash.
    ///
    /// Bounded because the real corpus is not small: corpus-v6 is **185,475
    /// shard files** against an **80 MB manifest**, 306.5M tokens. Reading all of
    /// it per step is not a performance nit, it is impossible. Only
    /// `shards_per_step` are read, chosen deterministically from `worker_shard`
    /// and `step` so two workers draw DIFFERENT shards — that is the
    /// data-parallel split, and without it every worker trains the same slice.
    fn read_and_verify_shards(
        &self,
        manifest: &CorpusManifest,
        step: StepIndex,
        worker_shard: u32,
    ) -> anyhow::Result<Vec<Shard>> {
        let total = manifest.shards.len();
        anyhow::ensure!(total > 0, "corpus manifest declares no shards");

        let want = self.params.shards_per_step.min(total);
        // Deterministic, worker-disjoint stride. Same (step, shard) always picks
        // the same slice, so a challenger replaying the step reads what we read.
        let offset = ((worker_shard as usize)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((step as usize).wrapping_mul(total / want.max(1) + 1)))
            % total;

        let mut out = Vec::with_capacity(want);
        for i in 0..want {
            let idx = (offset + i) % total;
            let meta = &manifest.shards[idx];
            let path = self
                .artifacts
                .manifest_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("manifest has no parent directory"))?
                .join(format!("shard_{:04}.json", meta.shard_index));

            let raw = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("shard {} unreadable at {}: {e}", meta.shard_index, path.display()))?;
            let shard: Shard = serde_json::from_str(&raw)?;

            if ShardManifest::of(&shard).provenance_root != meta.provenance_root {
                anyhow::bail!(
                    "shard {} does not reproduce the provenance_root the verified \
                     manifest commits to. Refusing: these are not the documents \
                     dataset_hash names.",
                    meta.shard_index
                );
            }
            out.push(shard);
        }
        Ok(out)
    }
}

#[async_trait]
impl ModelBackend for NatBackend {
    /// TRUE — and every clause is discharged elsewhere, not asserted here:
    /// the weights and corpus are the ones the chain named (verified by
    /// `ArtifactStore::resolve`, which `JobArtifacts` cannot exist without), the
    /// shard documents reproduce the manifest's provenance roots, and the
    /// architecture came from the verified sidecar with anything unsupported
    /// refused by name before this point.
    fn honors_job_spec(&self) -> bool {
        true
    }

    /// Load the verified checkpoint.
    ///
    /// Rejects a hash other than the one these artifacts were resolved for: that
    /// would mean the caller is driving this backend for a different job than it
    /// verified, which is exactly the substitution verification exists to stop.
    async fn load_starting_weights(
        &self,
        model_start_hash: WeightsHash,
    ) -> anyhow::Result<WeightsHash> {
        if model_start_hash != self.artifacts.model_start_hash {
            anyhow::bail!(
                "asked to load {model_start_hash:?} but this backend verified {:?}. \
                 Refusing: a backend must not train weights it did not check.",
                self.artifacts.model_start_hash
            );
        }
        self.model.lock().load(&self.artifacts.checkpoint_dir)?;
        Ok(model_start_hash)
    }

    /// One training pass against the real corpus, reported as per-zone weight
    /// deltas.
    ///
    /// `worker_shard` seeds the shuffle so two workers on the same job draw
    /// different windows — that is the data-parallel split. Without it every
    /// worker computes the same delta and the gather aggregates one sample N
    /// times while looking perfectly healthy.
    async fn forward_backward(
        &self,
        _prev_weights: PrevWeightsHash,
        epoch: EpochIndex,
        step: StepIndex,
        worker_shard: u32,
    ) -> anyhow::Result<StepResult> {
        let manifest = self.load_manifest()?;
        let shards = self.read_and_verify_shards(&manifest, step, worker_shard)?;
        let shuffle_seed =
            ((epoch as u64) << 40) | ((step as u64) << 16) | worker_shard as u64;

        // Snapshot -> train -> snapshot, entirely in memory. NAT ADR-0011's
        // `named_parameters` removed the safetensors round-trip that used to sit
        // on the hot path of every step.
        let (pre, post) = {
            let mut model = self.model.lock();

            // Rebuilt every step, deliberately. The shards differ per step and
            // per worker (that is the data-parallel split), so caching windows
            // would train every step on the same slice while looking busy.
            let (ids, _targets) = nat_candle::corpus::next_byte_windows(
                &shards,
                self.artifacts.shape.seq_len,
                self.params.max_windows,
                model.device(),
            )?;

            let pre = model.named_parameters()?;
            model.train_one_pass(&ids, &self.params, shuffle_seed)?;
            let post = model.named_parameters()?;
            (pre, post)
        };

        let deltas = zone_deltas(&pre, &post)?;

        // Zone -> layer_index. `zone_deltas` returns zones sorted, so the index is
        // deterministic across workers; binding it here keeps the mapping visible
        // rather than implied by iteration order.
        let gradients = deltas
            .iter()
            .enumerate()
            .map(|(i, d)| Tensor {
                data: d.values.clone(),
                layer_index: i,
            })
            .collect();

        // The post-state rides the SAME grid as the commitment, so "what the
        // weights became" and "what was committed" are one arithmetic. Splitting
        // them across grids would make the two answers incomparable for a
        // challenger replaying the step.
        let post_weights_hash = match self.artifacts.commitment_grid {
            CommitmentGrid::Q16 => q16_step_commitment(
                &post
                    .iter()
                    .enumerate()
                    .map(|(i, (_, v))| (i as u32, to_q16(v)))
                    .collect::<Vec<_>>(),
            ),
            CommitmentGrid::LegacyF32Scale => {
                let commits: Vec<_> = post
                    .iter()
                    .map(|(_, v)| tensor_commitment(&quantize_tensor(v)))
                    .collect();
                crate::quantize::step_commitment(&commits)
            }
        };

        Ok(StepResult {
            gradients,
            post_weights_hash,
        })
    }

    /// Commit on the grid the JOB declared — not a grid this worker prefers.
    ///
    /// Reading it from the verified sidecar is the whole mechanism: the
    /// challenger reads the same field, so the two cannot disagree. A worker that
    /// hardcoded Q16 would be just as wrong as one that hardcoded the legacy
    /// path, for a job that said otherwise.
    ///
    /// `Q16` is the safe grid and the default. `LegacyF32Scale` rides
    /// `quantize_tensor`'s data-dependent f32 scale, under which two honest
    /// workers on different hardware can commit differently for the same result
    /// and one gets slashed 10% — see `q16_commitment` for both demonstrated
    /// failure forms. It exists so a job pinned to it is still trainable, not
    /// because it is a reasonable choice.
    fn compute_step_commitment(&self, gradients: &[Tensor]) -> CommitmentHash {
        match self.artifacts.commitment_grid {
            CommitmentGrid::Q16 => {
                let q: Vec<(u32, _)> = gradients
                    .iter()
                    .map(|t| (t.layer_index as u32, to_q16(&t.data)))
                    .collect();
                q16_step_commitment(&q)
            }
            CommitmentGrid::LegacyF32Scale => {
                // Byte-for-byte the crate default, so a job pinned to the legacy
                // grid commits exactly what a default-backend worker would.
                let mut ordered: Vec<&Tensor> = gradients.iter().collect();
                ordered.sort_by_key(|t| t.layer_index);
                let commits: Vec<_> = ordered
                    .iter()
                    .map(|t| tensor_commitment(&quantize_tensor(&t.data)))
                    .collect();
                crate::quantize::step_commitment(&commits)
            }
        }
    }
}
