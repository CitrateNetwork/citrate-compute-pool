//! Running one `lora_delta` job on a member's device.
//!
//! Everything that can be refused is refused before the trainer starts, in this
//! order, each with a reason the coordinator log and the member can act on:
//!
//! 1. the payload parses and its round id matches its config and ordinal;
//! 2. this device is on the round's roster;
//! 3. the member consented to this round (D-29: per-round, explicit);
//! 4. this device has not already contributed to the round;
//! 5. a trainer is configured;
//! 6. the base model and start adapter are staged locally and hash-verify;
//! 7. the dataset is a verified trajectory export (every line carries the
//!    verifiers that passed it) and is not empty.
//!
//! After training, the trained adapter must have exactly the start adapter's
//! layout; the delta is computed, bounded, encoded, committed and signed.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::delta::{self, Artifact};
use super::gguf::Adapter;
use super::round::{LoraDeltaPayload, LoraDeltaResult, TASK_LORA_DELTA};
use super::trainer::{LoraTrainer, TrainRequest};
use super::{hex0x, sha256, sha256_file, Addr, B32};
use crate::coordinator_protocol::JobSpec;
use crate::wallet::Wallet;

/// Largest dataset a device will hash and hand to its trainer (placeholder,
/// pending owner sign-off).
pub const MAX_DATASET_BYTES: u64 = 256 << 20;

#[derive(Debug, thiserror::Error)]
pub enum LoraRunError {
    #[error("payload is not a lora_delta job: {0}")]
    BadPayload(String),
    #[error("this device ({0}) is not on the round's roster")]
    NotInRoster(String),
    #[error(
        "the member has not consented to round {0}. Add it to the file named by \
         CITRATE_FL_CONSENT_FILE to take part (D-29: consent is per round)"
    )]
    NoConsent(String),
    #[error("this device already contributed to round {0}; leaving the job to another device")]
    AlreadyContributed(String),
    #[error(
        "no LoRA trainer is configured on this device (set CITRATE_LORA_TRAINER to the \
         operator's trainer program); declining so another device takes the job"
    )]
    NoTrainer,
    #[error("{what} is not staged at {path}: {detail}")]
    NotStaged {
        what: &'static str,
        path: String,
        detail: String,
    },
    #[error("{what} at {path} has sha256 {got}, the round names {want}")]
    HashMismatch {
        what: &'static str,
        path: String,
        got: String,
        want: String,
    },
    #[error("dataset: {0}")]
    Dataset(String),
    #[error("training: {0}")]
    Training(String),
    #[error("adapter: {0}")]
    Adapter(String),
    #[error("delta: {0}")]
    Delta(String),
    #[error("signing: {0}")]
    Signing(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// What a successful run hands back: the signed result to submit and the
/// artifact to upload first.
#[derive(Debug, Clone)]
pub struct LoraRunOutput {
    pub result: LoraDeltaResult,
    pub result_json: String,
    pub artifact: Vec<u8>,
    pub artifact_sha256: B32,
}

#[derive(Deserialize)]
struct ConsentFile {
    rounds: Vec<String>,
}

/// One line of the S9.3 export (`citrate-agent-trajectory`), read for shape only.
#[derive(Deserialize)]
struct ExportLine {
    messages: Vec<ExportMessage>,
    metadata: ExportMeta,
}

#[derive(Deserialize)]
struct ExportMessage {
    role: String,
    #[allow(dead_code)]
    content: String,
}

#[derive(Deserialize)]
struct ExportMeta {
    #[allow(dead_code)]
    model: String,
    verifiers: Vec<String>,
}

/// Check a dataset is a verified trajectory export. Returns its example count.
pub fn verify_dataset(bytes: &[u8]) -> Result<u64, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "not UTF-8".to_string())?;
    let mut n = 0u64;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let ex: ExportLine = serde_json::from_str(line)
            .map_err(|e| format!("line {}: not a trajectory export line: {e}", i + 1))?;
        if ex.messages.is_empty() {
            return Err(format!("line {}: no messages", i + 1));
        }
        if let Some(m) = ex
            .messages
            .iter()
            .find(|m| !matches!(m.role.as_str(), "user" | "assistant" | "tool"))
        {
            return Err(format!(
                "line {}: role {:?} (the export never carries a system prompt)",
                i + 1,
                m.role
            ));
        }
        if ex.metadata.verifiers.is_empty() {
            return Err(format!(
                "line {}: no verifier passed this turn; only verified turns may be trained on",
                i + 1
            ));
        }
        n += 1;
    }
    if n == 0 {
        return Err("the export has no examples".into());
    }
    Ok(n)
}

/// Device-side configuration (member/operator controlled, never job controlled).
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// `<store>/models/<sha256>.gguf`, `<store>/adapters/<sha256>.gguf`,
    /// `<store>/contributed/`, `<store>/scratch/`.
    pub store: PathBuf,
    pub dataset: PathBuf,
    pub consent_file: Option<PathBuf>,
}

impl DeviceConfig {
    /// `CITRATE_FL_STORE` (default `./fl`), `CITRATE_FL_DATASET`,
    /// `CITRATE_FL_CONSENT_FILE`.
    pub fn from_env() -> Option<Self> {
        let dataset = std::env::var("CITRATE_FL_DATASET").ok()?;
        Some(Self {
            store: std::env::var("CITRATE_FL_STORE")
                .unwrap_or_else(|_| "./fl".into())
                .into(),
            dataset: dataset.into(),
            consent_file: std::env::var("CITRATE_FL_CONSENT_FILE")
                .ok()
                .map(Into::into),
        })
    }

    pub fn model_path(&self, sha: &B32) -> PathBuf {
        self.store
            .join("models")
            .join(format!("{}.gguf", hex::encode(sha)))
    }

    pub fn adapter_path(&self, sha: &B32) -> PathBuf {
        self.store
            .join("adapters")
            .join(format!("{}.gguf", hex::encode(sha)))
    }

    fn marker(&self, round: &B32) -> PathBuf {
        self.store.join("contributed").join(hex::encode(round))
    }
}

pub struct LoraDeltaRunner {
    device: DeviceConfig,
    trainer: Option<Box<dyn LoraTrainer>>,
    wallet: Wallet,
}

fn verify_file(what: &'static str, path: &Path, want: &B32) -> Result<(), LoraRunError> {
    let got = sha256_file(path).map_err(|e| LoraRunError::NotStaged {
        what,
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    if &got != want {
        return Err(LoraRunError::HashMismatch {
            what,
            path: path.display().to_string(),
            got: hex::encode(got),
            want: hex::encode(want),
        });
    }
    Ok(())
}

impl LoraDeltaRunner {
    pub fn new(
        device: DeviceConfig,
        trainer: Option<Box<dyn LoraTrainer>>,
        wallet: Wallet,
    ) -> Self {
        Self {
            device,
            trainer,
            wallet,
        }
    }

    fn me(&self) -> Addr {
        self.wallet.address().to_fixed_bytes()
    }

    fn consented(&self, round: &B32) -> Result<bool, LoraRunError> {
        let Some(path) = &self.device.consent_file else {
            return Ok(false);
        };
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let f: ConsentFile = serde_json::from_str(&raw)
            .map_err(|e| LoraRunError::BadPayload(format!("consent file: {e}")))?;
        let want = hex0x(round);
        Ok(f.rounds
            .iter()
            .any(|r| r.trim().eq_ignore_ascii_case(&want)))
    }

    /// Record that this device contributed to `round`, so a second job of the
    /// same round is left for another device. Called once the delta is uploaded.
    pub fn mark_contributed(&self, round: &B32) -> Result<(), LoraRunError> {
        let m = self.device.marker(round);
        if let Some(dir) = m.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(m, b"")?;
        Ok(())
    }

    /// True when this job is a federated LoRA job this runner should handle.
    pub fn handles(job: &JobSpec) -> bool {
        job.payload.get("task").and_then(|t| t.as_str()) == Some(TASK_LORA_DELTA)
    }

    pub async fn run(&self, job: &JobSpec) -> Result<LoraRunOutput, LoraRunError> {
        let payload: LoraDeltaPayload = serde_json::from_value(job.payload.clone())
            .map_err(|e| LoraRunError::BadPayload(e.to_string()))?;
        payload
            .validate()
            .map_err(|e| LoraRunError::BadPayload(e.to_string()))?;
        let cfg = &payload.config;
        let me = self.me();
        if !cfg.in_roster(&me) {
            return Err(LoraRunError::NotInRoster(hex0x(&me)));
        }
        let round_hex = hex0x(&payload.round_id);
        if !self.consented(&payload.round_id)? {
            return Err(LoraRunError::NoConsent(round_hex));
        }
        if self.device.marker(&payload.round_id).exists() {
            return Err(LoraRunError::AlreadyContributed(round_hex));
        }
        let trainer = self.trainer.as_ref().ok_or(LoraRunError::NoTrainer)?;

        let base = self.device.model_path(&cfg.base_model_sha256);
        verify_file("the base model", &base, &cfg.base_model_sha256)?;
        let start_path = self.device.adapter_path(&cfg.start_adapter_sha256);
        verify_file("the start adapter", &start_path, &cfg.start_adapter_sha256)?;

        let meta =
            std::fs::metadata(&self.device.dataset).map_err(|e| LoraRunError::NotStaged {
                what: "the trajectory export",
                path: self.device.dataset.display().to_string(),
                detail: e.to_string(),
            })?;
        if meta.len() > MAX_DATASET_BYTES {
            return Err(LoraRunError::Dataset(format!(
                "{} bytes exceeds the {MAX_DATASET_BYTES}-byte limit",
                meta.len()
            )));
        }
        let data = std::fs::read(&self.device.dataset)?;
        let examples = verify_dataset(&data).map_err(LoraRunError::Dataset)?;
        let dataset_sha256 = sha256(&data);

        let scratch = self.device.store.join("scratch");
        std::fs::create_dir_all(&scratch)?;
        let out = scratch.join(format!("{}-trained.gguf", hex::encode(payload.round_id)));
        let started = std::time::Instant::now();
        trainer
            .train(&TrainRequest {
                base_model: base,
                start_adapter: start_path.clone(),
                dataset: self.device.dataset.clone(),
                out: out.clone(),
                round_id_hex: round_hex.clone(),
            })
            .await
            .map_err(|e| LoraRunError::Training(e.to_string()))?;

        let start = Adapter::load(&start_path).map_err(|e| LoraRunError::Adapter(e.to_string()))?;
        let trained = Adapter::load(&out).map_err(|e| LoraRunError::Adapter(e.to_string()))?;
        let manifest = delta::manifest_hash(&start);
        if delta::manifest_hash(&trained) != manifest {
            return Err(LoraRunError::Adapter(
                "the trained adapter's tensors differ from the start adapter's".into(),
            ));
        }
        let trained_sha = sha256_file(&out)?;
        let values = delta::compute(&start, &trained, cfg.value_scale_log2)
            .map_err(|e| LoraRunError::Delta(e.to_string()))?;
        if values.len() as u64 > cfg.max_values {
            return Err(LoraRunError::Delta(format!(
                "{} values exceed the round's limit of {}",
                values.len(),
                cfg.max_values
            )));
        }
        let delta_root = delta::delta_root(&values, cfg.chunk_dim)
            .map_err(|e| LoraRunError::Delta(e.to_string()))?;
        let artifact = Artifact {
            value_scale_log2: cfg.value_scale_log2,
            round_id: payload.round_id,
            worker: me,
            start_adapter_sha256: cfg.start_adapter_sha256,
            trained_adapter_sha256: trained_sha,
            manifest_hash: manifest,
            chunk_dim: cfg.chunk_dim,
            values,
        };
        let bytes = artifact.encode();
        let artifact_sha256 = sha256(&bytes);
        let mut result = LoraDeltaResult {
            task: TASK_LORA_DELTA.into(),
            round_id: payload.round_id,
            worker: me,
            delta_root,
            delta_sha256: artifact_sha256,
            n_values: artifact.values.len() as u64,
            chunk_dim: cfg.chunk_dim,
            start_adapter_sha256: cfg.start_adapter_sha256,
            trained_adapter_sha256: trained_sha,
            dataset_sha256,
            examples,
            trainer: trainer.id(),
            seconds: started.elapsed().as_secs_f64(),
            signature: String::new(),
        };
        let sig = self
            .wallet
            .sign_digest_recoverable(&result.digest())
            .map_err(|e| LoraRunError::Signing(e.to_string()))?;
        result.signature = hex0x(&sig);
        let result_json =
            serde_json::to_string(&result).map_err(|e| LoraRunError::Signing(e.to_string()))?;
        // The trained adapter stays on the device: only the delta leaves it.
        let _ = std::fs::remove_file(&out);
        Ok(LoraRunOutput {
            result,
            result_json,
            artifact: bytes,
            artifact_sha256,
        })
    }
}

#[cfg(test)]
mod tests {
    include!("runner_tests.rs");
}
