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
//! The cost is real and stated: a step is `save → train → save → diff`, so it is
//! I/O-heavy. A `varmap` accessor upstream would remove the round-trip.
//!
//! ## Thread confinement (not optional)
//!
//! `ModelBackend` requires `Send + Sync`. `AutoregLm` holds
//! `Vec<Box<dyn CausalCore>>` and that trait object carries no `Send` bound
//! upstream, so the TYPE is not `Send` — even though candle's tensors are. The
//! model therefore lives on a dedicated thread and is driven by commands.
//!
//! The alternative would have been `unsafe impl Send`, asserting a property of
//! someone else's private trait object. That assertion could silently become
//! false on any NAT bump, so it is not made. One line upstream
//! (`Box<dyn CausalCore + Send + Sync>`) removes the need for this thread.
//!
//! ## Both architectures, neither a fallback for the other
//!
//! [`Architecture::ZonePartitioned`] runs `AutoregLm` (causal attention HP/PF/CX,
//! causal SSM SM/CB; `MX` is the non-learned harness and is never trained).
//! [`Architecture::Dense`] runs `AutoregDenseLm`. Which one is decided by the
//! verified sidecar — and a job asking for anything else, mixture-of-experts
//! included, never reaches this module: `ArtifactStore::resolve` refuses it.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread;

use async_trait::async_trait;

use nat_candle::autoreg::{AutoregConfig, AutoregDenseLm, AutoregLm};
use nat_data::manifest::{CorpusManifest, Shard, ShardManifest};
use nat_types::ZoneId;

use crate::backend::{ModelBackend, StepResult, Tensor};
use crate::job_artifacts::{Architecture, JobArtifacts};
use crate::q16_commitment::{q16_step_commitment, to_q16};
use crate::types::{CommitmentHash, EpochIndex, PrevWeightsHash, StepIndex, WeightsHash};
use crate::zone_delta::zone_deltas;

/// Hyper-parameters for a step. Not read from chain: the job spec carries the
/// artifact hashes and `steps_per_epoch`, not the optimizer setup.
#[derive(Debug, Clone)]
pub struct TrainingParams {
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Windows drawn from the corpus. Bounds memory on a large corpus.
    pub max_windows: usize,
    pub seq_len: usize,
    /// Model width. With `seq_len` and the zone list this fixes the parameter
    /// count, so it must match the checkpoint being resumed or `load` fails.
    pub d: usize,
    /// Dense-arm FFN width. Ignored by the zone arm.
    pub d_ff: usize,
    pub seed: u64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            batch_size: 16,
            learning_rate: 1e-3,
            max_windows: 4096,
            seq_len: 64,
            d: 48,
            d_ff: 192,
            seed: 2026,
        }
    }
}

/// Where the shard documents sit relative to their manifest.
fn shards_path(manifest: &Path) -> PathBuf {
    let mut p = manifest.to_path_buf();
    p.set_extension("shards.json");
    p
}

/// Commands to the model thread. Every payload is plain data, so nothing
/// non-`Send` crosses the channel.
enum Cmd {
    Load {
        dir: PathBuf,
        reply: Sender<anyhow::Result<()>>,
    },
    Step {
        shards: Vec<Shard>,
        shuffle_seed: u64,
        before: PathBuf,
        after: PathBuf,
        reply: Sender<anyhow::Result<()>>,
    },
    BackendTag {
        reply: Sender<&'static str>,
    },
}

/// The model, in whichever arm the job asked for. Lives only on its thread.
enum Model {
    Zone(Box<AutoregLm>),
    Dense(Box<AutoregDenseLm>),
}

impl Model {
    fn save(&self, dir: &Path) -> anyhow::Result<()> {
        match self {
            Model::Zone(m) => m.save(dir)?,
            Model::Dense(m) => m.save(dir)?,
        }
        Ok(())
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

fn build_model(arch: Architecture, p: &TrainingParams) -> anyhow::Result<Model> {
    Ok(match arch {
        Architecture::ZonePartitioned => {
            let cfg = AutoregConfig {
                // The five LEARNED zones. `MX` is the non-learned executive
                // harness: `AutoregLm` bails on a zone with no core, and the
                // federated seam rejects an MX delta — so it is excluded here
                // rather than filtered downstream.
                zones: vec![ZoneId::SM, ZoneId::CB, ZoneId::HP, ZoneId::PF, ZoneId::CX],
                vocab: nat_data::tokenizer::BYTE_VOCAB,
                seq_len: p.seq_len,
                d: p.d,
                tau: 1.0,
                seed: p.seed,
            };
            Model::Zone(Box::new(AutoregLm::new(&cfg)?))
        }
        Architecture::Dense => Model::Dense(Box::new(AutoregDenseLm::new(
            nat_data::tokenizer::BYTE_VOCAB,
            p.seq_len,
            p.d,
            p.d_ff,
            p.seed,
        )?)),
    })
}

/// NAT-backed [`ModelBackend`].
pub struct NatBackend {
    artifacts: JobArtifacts,
    tx: Sender<Cmd>,
    scratch: PathBuf,
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
        scratch: impl Into<PathBuf>,
    ) -> anyhow::Result<Self> {
        let scratch = scratch.into();
        std::fs::create_dir_all(&scratch)?;

        let arch = artifacts.architecture;
        let (tx, rx) = mpsc::channel::<Cmd>();
        let (ready_tx, ready_rx) = mpsc::channel::<anyhow::Result<()>>();

        thread::Builder::new()
            .name("nat-model".into())
            .spawn(move || {
                let mut model = match build_model(arch, &params) {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // The corpus is identical every step; building it once and caching
                // avoids re-reading a large shard set per step.
                let mut windows: Option<candle_core::Tensor> = None;

                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Cmd::Load { dir, reply } => {
                            let _ = reply.send(model.load(&dir));
                        }
                        Cmd::BackendTag { reply } => {
                            let _ = reply.send(model.backend_tag());
                        }
                        Cmd::Step {
                            shards,
                            shuffle_seed,
                            before,
                            after,
                            reply,
                        } => {
                            let r = (|| -> anyhow::Result<()> {
                                if windows.is_none() {
                                    let (ids, _t) = nat_candle::corpus::next_byte_windows(
                                        &shards,
                                        params.seq_len,
                                        params.max_windows,
                                        model.device(),
                                    )?;
                                    windows = Some(ids);
                                }
                                let ids = windows.as_ref().expect("just set");
                                model.save(&before)?;
                                model.train_one_pass(ids, &params, shuffle_seed)?;
                                model.save(&after)?;
                                Ok(())
                            })();
                            let _ = reply.send(r);
                        }
                    }
                }
            })?;

        ready_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("model thread died during construction"))??;

        Ok(Self {
            artifacts,
            tx,
            scratch,
        })
    }

    /// `"candle-cpu"` or `"candle-cuda"`.
    pub fn backend_tag(&self) -> anyhow::Result<&'static str> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(Cmd::BackendTag { reply: tx })?;
        Ok(rx.recv()?)
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

    /// Load the shard documents and check each against the manifest's committed
    /// `provenance_root`.
    ///
    /// `CorpusManifest` is METADATA: its `shards` are per-shard counts, quality
    /// and a provenance root — not the documents. So `dataset_hash` verifying the
    /// manifest is necessary and NOT sufficient. This is the step that binds the
    /// bytes actually trained on to the on-chain hash.
    fn load_and_verify_shards(&self) -> anyhow::Result<Vec<Shard>> {
        let manifest = self.load_manifest()?;
        let path = shards_path(&self.artifacts.manifest_path);
        let raw = std::fs::read_to_string(&path).map_err(|e| {
            anyhow::anyhow!("shard documents not present at {}: {e}", path.display())
        })?;
        let shards: Vec<Shard> = serde_json::from_str(&raw)?;

        if shards.len() != manifest.shards.len() {
            anyhow::bail!(
                "corpus has {} shards but the verified manifest commits to {}",
                shards.len(),
                manifest.shards.len()
            );
        }
        for (shard, meta) in shards.iter().zip(manifest.shards.iter()) {
            if ShardManifest::of(shard).provenance_root != meta.provenance_root {
                anyhow::bail!(
                    "shard {} does not reproduce the provenance_root the verified \
                     manifest commits to. Refusing: these are not the documents \
                     dataset_hash names.",
                    meta.shard_index
                );
            }
        }
        Ok(shards)
    }

    /// Read a safetensors checkpoint as `(name, values)` — the input
    /// `zone_delta` attributes by name.
    fn read_checkpoint(dir: &Path) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
        let tensors =
            candle_core::safetensors::load(dir.join("model.safetensors"), &candle_core::Device::Cpu)?;
        let mut out = Vec::with_capacity(tensors.len());
        for (name, t) in tensors {
            out.push((name, t.flatten_all()?.to_vec1::<f32>()?));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
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
        let (tx, rx) = mpsc::channel();
        self.tx.send(Cmd::Load {
            dir: self.artifacts.checkpoint_dir.clone(),
            reply: tx,
        })?;
        rx.recv()??;
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
        let shards = self.load_and_verify_shards()?;

        let before = self.scratch.join("before");
        let after = self.scratch.join("after");
        std::fs::create_dir_all(&before)?;
        std::fs::create_dir_all(&after)?;

        let shuffle_seed =
            ((epoch as u64) << 40) | ((step as u64) << 16) | worker_shard as u64;

        let (tx, rx) = mpsc::channel();
        self.tx.send(Cmd::Step {
            shards,
            shuffle_seed,
            before: before.clone(),
            after: after.clone(),
            reply: tx,
        })?;
        rx.recv()??;

        let pre = Self::read_checkpoint(&before)?;
        let post = Self::read_checkpoint(&after)?;
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

        // The post-state, hashed on the SAME Q16 grid as the commitment — so
        // "what the weights became" and "what was committed" are one arithmetic.
        let post_weights_hash = q16_step_commitment(
            &post
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i as u32, to_q16(v)))
                .collect::<Vec<_>>(),
        );

        Ok(StepResult {
            gradients,
            post_weights_hash,
        })
    }

    /// Commit on the shared Q16 grid, overriding the crate default.
    ///
    /// The default rides `quantize_tensor`'s data-dependent f32 scale, under
    /// which two honest workers on different hardware can commit differently for
    /// the same result and one gets slashed 10%. See `q16_commitment` for the two
    /// demonstrated failure forms.
    fn compute_step_commitment(&self, gradients: &[Tensor]) -> CommitmentHash {
        let q: Vec<(u32, _)> = gradients
            .iter()
            .map(|t| (t.layer_index as u32, to_q16(&t.data)))
            .collect();
        q16_step_commitment(&q)
    }
}
