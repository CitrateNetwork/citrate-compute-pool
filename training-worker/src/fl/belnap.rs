//! The `0x0110` Belnap-FOUR aggregation wire format, as the round uses it.
//!
//! Input (big-endian): `dim:u32 ‖ n:u32 ‖ embeddings[n·dim]:i64 ‖
//! confidences[n·dim]:i64 ‖ weights[n]:i64 ‖ threshold_pos:i64 ‖
//! threshold_neg:i64`, participant-major (row `i` is participant `i`'s values for
//! this chunk). Output: `aggregated[dim]:i64 ‖ states[dim]:u8`.
//!
//! The precompile computes `Σ_i weight_i · embedding_i[d]` on the Q16 grid and a
//! Belnap state per coordinate. With uniform weights that is the federated mean
//! of the participants' deltas, and the state says whether they agreed on its
//! sign (`True`), disagreed (`Both`) or none moved it (`Neither`).
//!
//! This module never computes an aggregate: the coordinator gets the output from
//! the precompile on a Citrate node, and the chain's replay recomputes it with
//! the precompile's own kernel. Building the input here is deterministic and is
//! re-derived by the replay from the delta artifacts.

use serde::{Deserialize, Serialize};

/// Precompile caps (`core/execution/src/precompiles/q16/belnap.rs`).
pub const MAX_DIM: usize = 1024;
pub const MAX_N: usize = 1024;
/// One unit on the Q16 grid.
pub const Q16_ONE: i64 = 1 << 16;

/// How each coordinate's confidence is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceRule {
    /// Confidence 1.0 where the participant moved the coordinate, 0 where its
    /// delta is exactly zero. A coordinate nobody moved is `Neither`.
    Nonzero,
}

impl ConfidenceRule {
    pub fn code(self) -> u8 {
        match self {
            ConfidenceRule::Nonzero => 1,
        }
    }

    pub fn confidence(self, value: i64) -> i64 {
        match self {
            ConfidenceRule::Nonzero => {
                if value == 0 {
                    0
                } else {
                    Q16_ONE
                }
            }
        }
    }
}

/// How participants are weighted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightRule {
    /// Every participant weighs `floor(1.0 / n)` on the Q16 grid.
    Uniform,
}

impl WeightRule {
    pub fn code(self) -> u8 {
        match self {
            WeightRule::Uniform => 1,
        }
    }

    pub fn weights(self, n: usize) -> Vec<i64> {
        match self {
            WeightRule::Uniform => vec![Q16_ONE / n.max(1) as i64; n],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BelnapWireError {
    #[error("chunk has {0} participants; the precompile takes 1..={MAX_N}")]
    Participants(usize),
    #[error("chunk is {0} wide; the precompile takes 1..={MAX_DIM}")]
    Dim(usize),
    #[error("participant {index} has {got} values, expected {dim}")]
    Ragged {
        index: usize,
        got: usize,
        dim: usize,
    },
    #[error("output is {got} bytes, expected {want} for a {dim}-wide chunk")]
    OutputLength { got: usize, want: usize, dim: usize },
    #[error("output state byte {0} is not a Belnap state")]
    State(u8),
}

/// The rules a round fixes for every chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRules {
    pub confidence: ConfidenceRule,
    pub weight: WeightRule,
    pub threshold_pos: i64,
    pub threshold_neg: i64,
}

/// Build the precompile input for one chunk. `rows[i]` is participant `i`'s
/// values for this chunk, participants in the round's canonical order.
pub fn chunk_input(rows: &[&[i64]], rules: &ChunkRules) -> Result<Vec<u8>, BelnapWireError> {
    let n = rows.len();
    if n == 0 || n > MAX_N {
        return Err(BelnapWireError::Participants(n));
    }
    let dim = rows[0].len();
    if dim == 0 || dim > MAX_DIM {
        return Err(BelnapWireError::Dim(dim));
    }
    for (index, r) in rows.iter().enumerate() {
        if r.len() != dim {
            return Err(BelnapWireError::Ragged {
                index,
                got: r.len(),
                dim,
            });
        }
    }
    let mut out = Vec::with_capacity(24 + 16 * n * dim + 8 * n);
    out.extend_from_slice(&(dim as u32).to_be_bytes());
    out.extend_from_slice(&(n as u32).to_be_bytes());
    for r in rows {
        for v in *r {
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
    for r in rows {
        for v in *r {
            out.extend_from_slice(&rules.confidence.confidence(*v).to_be_bytes());
        }
    }
    for w in rules.weight.weights(n) {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out.extend_from_slice(&rules.threshold_pos.to_be_bytes());
    out.extend_from_slice(&rules.threshold_neg.to_be_bytes());
    Ok(out)
}

/// A decoded precompile output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkOutput {
    /// Aggregated values, Q16 raw.
    pub values: Vec<i64>,
    /// Belnap states: 0 Neither, 1 True, 2 False, 3 Both.
    pub states: Vec<u8>,
}

pub fn parse_output(bytes: &[u8], dim: usize) -> Result<ChunkOutput, BelnapWireError> {
    let want = 9 * dim;
    if bytes.len() != want || dim == 0 {
        return Err(BelnapWireError::OutputLength {
            got: bytes.len(),
            want,
            dim,
        });
    }
    let (vals, states) = bytes.split_at(8 * dim);
    let values = vals
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_be_bytes(*c))
        .collect();
    for s in states {
        if *s > 3 {
            return Err(BelnapWireError::State(*s));
        }
    }
    Ok(ChunkOutput {
        values,
        states: states.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    include!("belnap_tests.rs");
}
