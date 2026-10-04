//! Round configuration, the `lora_delta` job payload and the signed worker
//! result (FL_ROUND_V1 §2–§3).
//!
//! The config is cluster-scoped: it names the ledger and cluster it belongs to,
//! the roster of device keys that may contribute, and every rule the aggregate
//! depends on. Its hash is what the cluster registers on chain and what every
//! round record snapshots, so a rule cannot change under a round in flight.

use serde::{Deserialize, Serialize};

use super::belnap::{ChunkRules, ConfidenceRule, WeightRule, MAX_DIM};
use super::{hexser, keccak, Addr, B32};

pub const TASK_LORA_DELTA: &str = "lora_delta";
pub const CONFIG_DOMAIN: &[u8] = b"citrate-fl-round-config/1";
pub const ROUND_DOMAIN: &[u8] = b"citrate-fl-round-key/1";
pub const DELTA_DOMAIN: &[u8] = b"citrate-fl-delta/1";

/// D-29 / US-9.1: a federated round has at least three devices.
pub const MIN_PARTICIPANTS_FLOOR: u16 = 3;
/// A fraud proof carries one chunk's whole precompile input in calldata, so
/// `participants × chunk_dim` is bounded to keep it a transaction a member can
/// afford to send (FL_ROUND_V1 §5). Placeholder, pending owner sign-off.
pub const MAX_CHUNK_CELLS: u64 = 4096;
/// Largest scale exponent; 2^16 on top of the Q16 grid leaves 31 bits of
/// headroom in an i64 for any finite f32 delta below 2^16.
pub const MAX_SCALE_LOG2: u8 = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundConfig {
    pub chain_id: u64,
    #[serde(with = "hexser")]
    pub ledger: Addr,
    #[serde(with = "hexser")]
    pub cluster_id: B32,
    /// SHA-256 of the base model GGUF the adapter applies to.
    #[serde(with = "hexser")]
    pub base_model_sha256: B32,
    /// SHA-256 of the adapter every participant starts from.
    #[serde(with = "hexser")]
    pub start_adapter_sha256: B32,
    /// Device keys that may contribute, strictly ascending.
    #[serde(with = "hexser::vec20")]
    pub roster: Vec<Addr>,
    pub min_participants: u16,
    pub chunk_dim: u32,
    /// Deltas are multiplied by `2^value_scale_log2` before Q16 encoding, so
    /// small LoRA updates keep their precision on the fixed grid.
    pub value_scale_log2: u8,
    pub threshold_pos: i64,
    pub threshold_neg: i64,
    pub confidence: ConfidenceRule,
    pub weight: WeightRule,
    /// Largest delta (in values) a participant may submit.
    pub max_values: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error("the roster must be strictly ascending (sorted, no duplicates)")]
    RosterOrder,
    #[error("min_participants {0} is below the floor of {MIN_PARTICIPANTS_FLOOR}")]
    MinParticipants(u16),
    #[error("the roster has {roster} devices, fewer than min_participants {min}")]
    RosterTooSmall { roster: usize, min: u16 },
    #[error("chunk_dim {0} must be 1..={MAX_DIM}")]
    ChunkDim(u32),
    #[error("roster {roster} x chunk_dim {chunk} exceeds {MAX_CHUNK_CELLS} cells per chunk")]
    ChunkCells { roster: usize, chunk: u32 },
    #[error("value_scale_log2 {0} exceeds {MAX_SCALE_LOG2}")]
    Scale(u8),
    #[error("threshold_pos must be positive and threshold_neg negative")]
    Thresholds,
    #[error("max_values must be positive")]
    MaxValues,
    #[error("{0} is zero")]
    Zero(&'static str),
}

impl RoundConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cluster_id == [0u8; 32] {
            return Err(ConfigError::Zero("cluster_id"));
        }
        if self.ledger == [0u8; 20] {
            return Err(ConfigError::Zero("ledger"));
        }
        if self.start_adapter_sha256 == [0u8; 32] {
            return Err(ConfigError::Zero("start_adapter_sha256"));
        }
        if self.base_model_sha256 == [0u8; 32] {
            return Err(ConfigError::Zero("base_model_sha256"));
        }
        if !self.roster.windows(2).all(|w| w[0] < w[1]) {
            return Err(ConfigError::RosterOrder);
        }
        if self.roster.contains(&[0u8; 20]) {
            return Err(ConfigError::Zero("a roster address"));
        }
        if self.min_participants < MIN_PARTICIPANTS_FLOOR {
            return Err(ConfigError::MinParticipants(self.min_participants));
        }
        if self.roster.len() < usize::from(self.min_participants) {
            return Err(ConfigError::RosterTooSmall {
                roster: self.roster.len(),
                min: self.min_participants,
            });
        }
        if self.chunk_dim == 0 || self.chunk_dim as usize > MAX_DIM {
            return Err(ConfigError::ChunkDim(self.chunk_dim));
        }
        if (self.roster.len() as u64).saturating_mul(u64::from(self.chunk_dim)) > MAX_CHUNK_CELLS {
            return Err(ConfigError::ChunkCells {
                roster: self.roster.len(),
                chunk: self.chunk_dim,
            });
        }
        if self.value_scale_log2 > MAX_SCALE_LOG2 {
            return Err(ConfigError::Scale(self.value_scale_log2));
        }
        if self.threshold_pos <= 0 || self.threshold_neg >= 0 {
            return Err(ConfigError::Thresholds);
        }
        if self.max_values == 0 {
            return Err(ConfigError::MaxValues);
        }
        Ok(())
    }

    /// `keccak256` over the packed fields (FL_ROUND_V1 §2). Registered on chain
    /// for the cluster and snapshotted into every round record.
    pub fn config_hash(&self) -> B32 {
        let roster: Vec<u8> = self.roster.iter().flatten().copied().collect();
        keccak(&[
            CONFIG_DOMAIN,
            &self.chain_id.to_be_bytes(),
            &self.ledger,
            &self.cluster_id,
            &self.base_model_sha256,
            &self.start_adapter_sha256,
            &self.min_participants.to_be_bytes(),
            &self.chunk_dim.to_be_bytes(),
            &[self.value_scale_log2],
            &self.threshold_pos.to_be_bytes(),
            &self.threshold_neg.to_be_bytes(),
            &[self.confidence.code(), self.weight.code()],
            &self.max_values.to_be_bytes(),
            &(self.roster.len() as u32).to_be_bytes(),
            &roster,
        ])
    }

    /// The on-chain round id for `ordinal`; `FederatedRoundLedger.roundIdOf`
    /// computes the same value.
    pub fn round_id(&self, ordinal: u64) -> B32 {
        round_id(self.chain_id, &self.ledger, &self.cluster_id, ordinal)
    }

    pub fn chunk_rules(&self) -> ChunkRules {
        ChunkRules {
            confidence: self.confidence,
            weight: self.weight,
            threshold_pos: self.threshold_pos,
            threshold_neg: self.threshold_neg,
        }
    }

    pub fn in_roster(&self, who: &Addr) -> bool {
        self.roster.binary_search(who).is_ok()
    }
}

pub fn round_id(chain_id: u64, ledger: &Addr, cluster_id: &B32, ordinal: u64) -> B32 {
    keccak(&[
        ROUND_DOMAIN,
        &chain_id.to_be_bytes(),
        ledger,
        cluster_id,
        &ordinal.to_be_bytes(),
    ])
}

/// What the coordinator hands a worker for one round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraDeltaPayload {
    /// Always [`TASK_LORA_DELTA`].
    pub task: String,
    pub ordinal: u64,
    #[serde(with = "hexser")]
    pub round_id: B32,
    pub config: RoundConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PayloadError {
    #[error("not a lora_delta job: task {0:?}")]
    Task(String),
    #[error("round config: {0}")]
    Config(#[from] ConfigError),
    #[error("round_id does not match the config and ordinal")]
    RoundId,
}

impl LoraDeltaPayload {
    pub fn new(config: RoundConfig, ordinal: u64) -> Self {
        Self {
            task: TASK_LORA_DELTA.into(),
            ordinal,
            round_id: config.round_id(ordinal),
            config,
        }
    }

    /// The checks every consumer runs before trusting a payload.
    pub fn validate(&self) -> Result<(), PayloadError> {
        if self.task != TASK_LORA_DELTA {
            return Err(PayloadError::Task(self.task.clone()));
        }
        self.config.validate()?;
        if self.config.round_id(self.ordinal) != self.round_id {
            return Err(PayloadError::RoundId);
        }
        Ok(())
    }
}

/// The digest a worker signs over its delta. Binds the round, the worker, the
/// chunked commitment and the artifact bytes, so the round bundle can be
/// verified without trusting the coordinator that collected it.
pub fn delta_digest(
    round_id: &B32,
    worker: &Addr,
    delta_root: &B32,
    delta_sha256: &B32,
    n_values: u64,
    chunk_dim: u32,
) -> B32 {
    keccak(&[
        DELTA_DOMAIN,
        round_id,
        worker,
        delta_root,
        delta_sha256,
        &n_values.to_be_bytes(),
        &chunk_dim.to_be_bytes(),
    ])
}

/// What a worker returns for a `lora_delta` job (the submission payload).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraDeltaResult {
    /// Always [`TASK_LORA_DELTA`].
    pub task: String,
    #[serde(with = "hexser")]
    pub round_id: B32,
    #[serde(with = "hexser")]
    pub worker: Addr,
    #[serde(with = "hexser")]
    pub delta_root: B32,
    #[serde(with = "hexser")]
    pub delta_sha256: B32,
    pub n_values: u64,
    pub chunk_dim: u32,
    #[serde(with = "hexser")]
    pub start_adapter_sha256: B32,
    #[serde(with = "hexser")]
    pub trained_adapter_sha256: B32,
    /// SHA-256 of the verified trajectory export the worker trained on.
    #[serde(with = "hexser")]
    pub dataset_sha256: B32,
    /// Examples in that export.
    pub examples: u64,
    /// Which trainer produced the adapter (operator-configured program name and
    /// the SHA-256 of its executable), recorded as provenance.
    pub trainer: String,
    pub seconds: f64,
    /// Recoverable secp256k1 signature over [`delta_digest`], `0x` hex.
    pub signature: String,
}

impl LoraDeltaResult {
    pub fn digest(&self) -> B32 {
        delta_digest(
            &self.round_id,
            &self.worker,
            &self.delta_root,
            &self.delta_sha256,
            self.n_values,
            self.chunk_dim,
        )
    }
}

#[cfg(test)]
mod tests {
    include!("round_tests.rs");
}
