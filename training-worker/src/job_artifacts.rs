//! Resolving a training job's on-chain spec to REAL, VERIFIED artifacts.
//!
//! This is the honesty core of the training backend. `ModelBackend::honors_job_spec`
//! is allowed to answer `true` only because this module exists: it is what turns
//! `spec.model_start_hash` and `spec.dataset_hash` from decorative bytes into the
//! actual weights and the actual corpus, checked by content.
//!
//! ## Why this is a separate, dependency-light layer
//!
//! Deliberately no Candle, no CUDA, no NAT. Verification is hashing and parsing;
//! it does not need a GPU stack, and keeping it independent means the part that
//! decides *whether it is honest to train* is fully testable on any machine, in
//! CI, without a 300 MB dependency tree. The training itself is feature-gated
//! behind that; this is not.
//!
//! ## What the chain says vs. what a worker must prove
//!
//! `ComputePoolTraining` publishes a job as `(model_start_hash, dataset_hash, …)`.
//! Those are commitments, not content. A worker that trains *something* and commits
//! a Merkle root of its gradients is paid regardless — the protocol cannot see
//! which model or which data produced the hashes (see
//! `tests/refuses_to_earn_on_a_placeholder_backend.rs`). So the binding has to be
//! made here, before the first step:
//!
//!   - the checkpoint bytes must hash to `model_start_hash`;
//!   - the corpus manifest must hash to `dataset_hash`;
//!   - the architecture must be one this worker can actually train.
//!
//! Any of those failing is a REFUSAL, never a fallback. A worker that quietly
//! trained a different model would still be paid, which is precisely the failure
//! mode being designed out.

use std::path::{Path, PathBuf};

use sha3::{Digest, Keccak256};

use crate::types::B256;

/// The model architecture a job asks for.
///
/// Resolved from the NAT sidecar that accompanies the checkpoint, NOT guessed
/// from tensor shapes — a guess that lands on the wrong arm would train the wrong
/// thing and still commit valid-looking roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// NAT's zone-partitioned LM (`nat_candle::autoreg::AutoregLm`). The hidden
    /// width is split across declared zones; attention cores for HP/PF/CX, causal
    /// SSM recurrence for SM/CB, and `MX` is a non-learned harness that is not
    /// trained at all.
    ZonePartitioned,
    /// The dense per-position autoregressive LM
    /// (`nat_candle::autoreg::AutoregDenseLm`). Same embedding and readout as the
    /// zone arm, a single causal attention block plus FFN instead of zones — this
    /// is the H-01 equal-parameter baseline, and a perfectly good standalone
    /// training target in its own right.
    Dense,
}

impl Architecture {
    pub fn as_str(self) -> &'static str {
        match self {
            Architecture::ZonePartitioned => "zone-partitioned",
            Architecture::Dense => "dense",
        }
    }
}

/// Which quantization grid a job's step commitments ride.
///
/// ## Why this is declared per job rather than chosen per worker
///
/// `ComputePoolTraining` never recomputes a commitment. `challengeStep` checks
/// Merkle INCLUSION only, and resolution is a committee vote — so the grid is
/// off-chain policy, and no contract change is needed to move it.
///
/// What that leaves is the thing that actually matters: **the worker that commits
/// and the committee that recomputes must use the same grid.** If they disagree,
/// every honest worker looks dishonest and gets slashed 10%. So the grid is a
/// property of the JOB, read from the same verified sidecar by both sides, not a
/// local setting either one picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentGrid {
    /// The fixed Q16.16 grid from `citrate_fed_types` — the same one NAT and the
    /// chain's `0x0110` path use. Data-independent, so two workers on different
    /// hardware commit identically. **The default.**
    Q16,
    /// The legacy per-tensor f32 scale (`max|x| / 127`, scale hashed into the
    /// preimage).
    ///
    /// Retained only so a job can be pinned to it deliberately. It is not safe
    /// for heterogeneous workers: a one-ULP difference in the maximum element
    /// changes the commitment, and — worse — shifts OTHER coordinates' quantized
    /// values, because they all share the derived scale. Both failures are
    /// demonstrated in `q16_commitment`'s tests.
    LegacyF32Scale,
}

impl CommitmentGrid {
    pub fn as_str(self) -> &'static str {
        match self {
            CommitmentGrid::Q16 => "q16",
            CommitmentGrid::LegacyF32Scale => "legacy-f32-scale",
        }
    }
}

/// The model's shape, read from the sidecar rather than assumed.
///
/// A checkpoint does not carry `seq_len` (the causal masks are constants, not
/// parameters), and inferring `vocab`/`d` from tensor shapes would be guessing at
/// something the job already states. Guessing wrong does not fail loudly — it
/// fails as a shape mismatch on load, or worse, loads and trains the wrong model.
///
/// The real 64M NAT checkpoint is `d = 1183`, `vocab = 16384`, BF16, five learned
/// zones — none of which is `BYTE_VOCAB`, and none of which a backend should be
/// hardcoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShape {
    pub vocab: usize,
    pub d: usize,
    pub seq_len: usize,
    /// Dense-arm FFN width. Ignored by the zone arm.
    pub d_ff: usize,
    /// `"bf16"` or `"f32"`. The 64M checkpoint is BF16; loading it into an F32
    /// model is a dtype mismatch, not a silent widening.
    pub dtype: String,
}

impl Default for ModelShape {
    /// A small F32 byte-level model — the shape the crate's own tests use. NOT a
    /// stand-in for a real job: a real job declares its shape in the sidecar.
    fn default() -> Self {
        Self {
            vocab: 256,
            d: 48,
            seq_len: 64,
            d_ff: 192,
            dtype: "f32".into(),
        }
    }
}

/// Why a job could not be resolved to something this worker may honestly train.
///
/// Every variant is a refusal. There is deliberately no "close enough" path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    /// The artifact is not on this host. Not a failure of honesty — the worker
    /// simply has not fetched it yet — but still a refusal to train.
    Missing { what: &'static str, path: PathBuf },
    /// The bytes are present but do not hash to what the chain committed. This is
    /// the serious one: it means the worker would be training a DIFFERENT model or
    /// a DIFFERENT corpus than the job it is being paid for.
    HashMismatch {
        what: &'static str,
        expected: String,
        actual: String,
    },
    /// The sidecar could not be read or parsed, so the architecture is unknown.
    /// Unknown is refused rather than defaulted — defaulting to dense would
    /// silently train the H-01 baseline for a zone job.
    UnreadableSidecar(String),
    /// The job asks for an architecture this worker cannot train.
    ///
    /// Notably **mixture-of-experts**: NAT rejected learned expert routing in
    /// ADR-0001 ("loses interpretability") and `02_ARCHITECTURE.md` §11 makes
    /// "declared zone partitioning … versus learned-from-scratch expert routing"
    /// the novelty wedge. There is no MoE trainer in NAT — no gate network, no
    /// expert dispatch, no load-balancing loss, no capacity factor. Training an
    /// MoE job on the dense arm would produce a real-looking model that is not the
    /// requested architecture, so it is refused by name.
    UnsupportedArchitecture(String),
    /// The sidecar named a commitment grid this worker does not implement.
    /// Refused rather than defaulted: a worker silently on a different grid from
    /// the challenger is how an honest worker gets slashed.
    UnknownCommitmentGrid(String),
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactError::Missing { what, path } => {
                write!(f, "{what} not present on this host at {}", path.display())
            }
            ArtifactError::HashMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} does not match the on-chain commitment: job expects {expected}, \
                 the bytes on disk hash to {actual}. Refusing to train — committing \
                 epochs from this would be paid work on the wrong artifact."
            ),
            ArtifactError::UnreadableSidecar(e) => {
                write!(f, "cannot determine the job's architecture from its sidecar: {e}")
            }
            ArtifactError::UnknownCommitmentGrid(g) => write!(
                f,
                "commitment grid '{g}' is not implemented by this worker. Refusing \
                 rather than defaulting — a worker committing on a different grid \
                 from the challenger is indistinguishable from a dishonest one."
            ),
            ArtifactError::UnsupportedArchitecture(a) => write!(
                f,
                "architecture '{a}' is not trainable by this worker. Refusing rather \
                 than substituting a different architecture, which would train a real \
                 model that is not the one the job asked for."
            ),
        }
    }
}

impl std::error::Error for ArtifactError {}

/// A job's artifacts, resolved and VERIFIED against the on-chain commitments.
///
/// Holding one of these is the evidence that lets `honors_job_spec()` answer
/// `true`: it cannot be constructed without both hashes matching.
#[derive(Debug, Clone)]
pub struct JobArtifacts {
    /// Directory holding `model.safetensors` — what `nat_candle`'s
    /// `AutoregLm::load` / `AutoregDenseLm::load` read.
    pub checkpoint_dir: PathBuf,
    /// The `nat-data` shard manifest describing the corpus.
    pub manifest_path: PathBuf,
    pub architecture: Architecture,
    /// The grid this job's commitments ride. Both the worker and the challenger
    /// read it from here, so they cannot disagree.
    pub commitment_grid: CommitmentGrid,
    /// The model's shape, from the sidecar. Never inferred.
    pub shape: ModelShape,
    /// The verified hashes, retained so the provenance record can state exactly
    /// what was trained rather than restating what was requested.
    pub model_start_hash: B256,
    pub dataset_hash: B256,
}

/// Where a worker keeps fetched artifacts. Content-addressed by the on-chain
/// hash, so two jobs naming the same model share one copy and a corrupted fetch
/// cannot masquerade as a good one.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `<root>/models/<hex>/` — the directory `nat_candle` loads from.
    pub fn model_dir(&self, hash: &B256) -> PathBuf {
        self.root.join("models").join(hex_of(hash))
    }

    /// `<root>/datasets/<hex>/` — the corpus directory.
    ///
    /// Layout is `nat-data`'s real pipeline output, not one invented here:
    /// `manifest.json` plus one `shard_NNNN.json` per shard. corpus-v6 is 185,475
    /// shard files against an 80 MB manifest, which is why the shards are read
    /// on demand rather than slurped — see `NatBackend`.
    pub fn dataset_dir(&self, hash: &B256) -> PathBuf {
        self.root.join("datasets").join(hex_of(hash))
    }

    /// `<root>/datasets/<hex>/manifest.json` — a `nat_data::manifest::CorpusManifest`.
    /// This is the file `dataset_hash` commits to.
    pub fn manifest_path(&self, hash: &B256) -> PathBuf {
        self.dataset_dir(hash).join("manifest.json")
    }

    /// `<dir>/shard_NNNN.json` — one `nat_data::manifest::Shard`.
    pub fn shard_path(&self, hash: &B256, index: u32) -> PathBuf {
        self.dataset_dir(hash).join(format!("shard_{index:04}.json"))
    }

    /// `<root>/models/<hex>/sidecar.nat.json` — the zone graph. Its presence is
    /// what distinguishes a zone-partitioned job from a dense one.
    pub fn sidecar_path(&self, hash: &B256) -> PathBuf {
        self.model_dir(hash).join("sidecar.nat.json")
    }

    /// Resolve and VERIFY both artifacts for a job.
    ///
    /// Returns `Ok` only when the checkpoint bytes hash to `model_start_hash`, the
    /// manifest bytes hash to `dataset_hash`, and the architecture is trainable.
    /// There is no partial success: a caller holding `JobArtifacts` may train.
    pub fn resolve(
        &self,
        model_start_hash: &B256,
        dataset_hash: &B256,
    ) -> Result<JobArtifacts, ArtifactError> {
        let checkpoint_dir = self.model_dir(model_start_hash);
        let weights = checkpoint_dir.join("model.safetensors");
        verify_file("model checkpoint", &weights, model_start_hash)?;

        let manifest_path = self.manifest_path(dataset_hash);
        verify_file("dataset manifest", &manifest_path, dataset_hash)?;

        let sidecar = self.sidecar_path(model_start_hash);
        let architecture = read_architecture(&sidecar)?;
        let commitment_grid = read_commitment_grid(&sidecar)?;
        let shape = read_model_shape(&sidecar)?;

        Ok(JobArtifacts {
            checkpoint_dir,
            manifest_path,
            architecture,
            commitment_grid,
            shape,
            model_start_hash: *model_start_hash,
            dataset_hash: *dataset_hash,
        })
    }
}

/// Keccak-256 of a file's bytes, compared against the on-chain commitment.
///
/// Keccak (not SHA-256) so the digest is reproducible by an on-chain verifier
/// with no extra precompile — the same hash family the rest of the chain uses.
fn verify_file(what: &'static str, path: &Path, expected: &B256) -> Result<(), ArtifactError> {
    let bytes = std::fs::read(path).map_err(|_| ArtifactError::Missing {
        what,
        path: path.to_path_buf(),
    })?;
    let actual = keccak_of(&bytes);
    if &actual != expected {
        return Err(ArtifactError::HashMismatch {
            what,
            expected: hex_of(expected),
            actual: hex_of(&actual),
        });
    }
    Ok(())
}

/// Determine the architecture from the sidecar next to the checkpoint.
///
/// No sidecar means a plain dense checkpoint — that is a real, supported job
/// shape, not a fallback: `AutoregDenseLm` is a standalone trainer as well as the
/// H-01 baseline. A sidecar that names an architecture we cannot train is refused
/// by name rather than approximated.
fn read_architecture(sidecar: &Path) -> Result<Architecture, ArtifactError> {
    let raw = match std::fs::read_to_string(sidecar) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Architecture::Dense),
        Err(e) => return Err(ArtifactError::UnreadableSidecar(e.to_string())),
    };

    let doc: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| ArtifactError::UnreadableSidecar(e.to_string()))?;

    // An explicit architecture wins, so a sidecar can name something we must
    // refuse (MoE) instead of being read as "has zones, therefore zone-trainable".
    if let Some(kind) = doc.get("architecture").and_then(|v| v.as_str()) {
        return match kind {
            "zone-partitioned" | "nat-zone" => Ok(Architecture::ZonePartitioned),
            "dense" => Ok(Architecture::Dense),
            other => Err(ArtifactError::UnsupportedArchitecture(other.to_string())),
        };
    }

    // Otherwise infer from the zone graph: a NAT sidecar with declared zones is a
    // zone-partitioned model.
    match doc.get("zones").and_then(|z| z.as_array()) {
        Some(zones) if !zones.is_empty() => Ok(Architecture::ZonePartitioned),
        _ => Ok(Architecture::Dense),
    }
}

/// Read the job's commitment grid from its sidecar.
///
/// Absent means [`CommitmentGrid::Q16`]. That default is deliberate in both
/// directions: it is the correct grid, and choosing the unsafe one has to be an
/// explicit act that shows up in the sidecar where a reviewer can see it. There
/// are no legacy jobs on 40204 to preserve — `ComputePoolTraining.nextJobId` is
/// still 0 — so nothing is broken by defaulting to the right answer.
///
/// An UNRECOGNISED grid is refused rather than defaulted. Silently falling back
/// to Q16 for a job that asked for something else would put the worker and the
/// challenger on different grids, which is the exact failure this field exists
/// to prevent.
fn read_commitment_grid(sidecar: &Path) -> Result<CommitmentGrid, ArtifactError> {
    let raw = match std::fs::read_to_string(sidecar) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(CommitmentGrid::Q16),
        Err(e) => return Err(ArtifactError::UnreadableSidecar(e.to_string())),
    };
    let doc: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| ArtifactError::UnreadableSidecar(e.to_string()))?;

    match doc.get("commitment_grid").and_then(|v| v.as_str()) {
        None => Ok(CommitmentGrid::Q16),
        Some("q16") => Ok(CommitmentGrid::Q16),
        Some("legacy-f32-scale") => Ok(CommitmentGrid::LegacyF32Scale),
        Some(other) => Err(ArtifactError::UnknownCommitmentGrid(other.to_string())),
    }
}

/// Read the model shape from the sidecar, falling back to the small test shape
/// only when the sidecar is absent entirely.
///
/// A sidecar that IS present but omits a field gets the default for that field —
/// deliberate, so a minimal sidecar stays writable — but a real checkpoint will
/// simply fail to load if the declared shape is wrong, which is the loud failure
/// we want rather than training a differently-shaped model.
fn read_model_shape(sidecar: &Path) -> Result<ModelShape, ArtifactError> {
    let raw = match std::fs::read_to_string(sidecar) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ModelShape::default()),
        Err(e) => return Err(ArtifactError::UnreadableSidecar(e.to_string())),
    };
    let doc: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| ArtifactError::UnreadableSidecar(e.to_string()))?;
    let d = ModelShape::default();
    let num = |k: &str, fallback: usize| -> usize {
        doc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(fallback)
    };
    Ok(ModelShape {
        vocab: num("vocab", d.vocab),
        d: num("d", d.d),
        seq_len: num("seq_len", d.seq_len),
        d_ff: num("d_ff", d.d_ff),
        dtype: doc
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or(&d.dtype)
            .to_string(),
    })
}

fn keccak_of(bytes: &[u8]) -> B256 {
    let mut h = Keccak256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out);
    B256::from(arr)
}

fn hex_of(h: &B256) -> String {
    format!("0x{}", hex::encode(h.as_bytes()))
}

#[cfg(test)]
mod tests {
    include!("job_artifacts_tests.rs");
}
