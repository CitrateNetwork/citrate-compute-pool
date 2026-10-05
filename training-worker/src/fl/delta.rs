//! A participant's adapter delta on the Q16 grid, its artifact encoding and its
//! chunked commitment (FL_ROUND_V1 §3).
//!
//! The delta is `trained − start`, tensor by tensor in **name order** of the
//! start adapter, scaled by `2^value_scale_log2` (exact for an f32) and encoded
//! with the federation's shared Q16 kernel (`citrate_fed_types::Q16::from_f32`).
//! Name order makes the coordinate layout identical on every device; the
//! manifest hash pins it, so two workers whose adapters differ in shape cannot
//! be averaged coordinate-for-coordinate by accident.
//!
//! The values are cut into chunks of `chunk_dim`; each chunk's row hash is
//! `keccak256` of its big-endian i64 bytes, exactly the bytes that participant's
//! row occupies inside the chunk's `0x0110` input. The delta root is the
//! positional tree over those row hashes.

use citrate_fed_types::Q16;

use super::gguf::{Adapter, OutTensor};
use super::{keccak, tree, Addr, B32};

pub const MAGIC: [u8; 4] = *b"FLD1";
pub const VERSION: u16 = 1;
pub const HEADER_LEN: usize = 4 + 2 + 1 + 1 + 32 + 20 + 32 + 32 + 32 + 4 + 8;
const MANIFEST_DOMAIN: &[u8] = b"citrate-fl-manifest/1";
/// A scaled delta beyond this magnitude is not a LoRA update; refuse it rather
/// than let the Q16 kernel saturate it silently.
pub const MAX_SCALED_ABS: f32 = 1_073_741_824.0; // 2^30

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DeltaError {
    #[error("adapters differ: {0}")]
    Mismatch(String),
    #[error("tensor {tensor:?} element {index} is not finite in the trained adapter")]
    NotFinite { tensor: String, index: usize },
    #[error("tensor {tensor:?} element {index}: scaled delta {value} is out of range")]
    OutOfRange {
        tensor: String,
        index: usize,
        value: f32,
    },
    #[error("artifact: {0}")]
    Artifact(String),
    #[error("chunk_dim must be positive")]
    ChunkDim,
    #[error("{0} values exceed the round's limit of {1}")]
    TooLarge(u64, u64),
}

/// The tensors of an adapter in canonical (name) order: `(file index, name, dims)`.
fn canonical(a: &Adapter) -> Vec<(usize, &str, &[u64])> {
    let mut v: Vec<(usize, &str, &[u64])> = a
        .header
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t.name.as_str(), t.dims.as_slice()))
        .collect();
    v.sort_by(|x, y| x.1.cmp(y.1));
    v
}

/// The layout every participant must share.
pub fn manifest_hash(a: &Adapter) -> B32 {
    let mut pre = Vec::new();
    pre.extend_from_slice(MANIFEST_DOMAIN);
    let c = canonical(a);
    pre.extend_from_slice(&(c.len() as u32).to_be_bytes());
    for (_, name, dims) in c {
        pre.extend_from_slice(&(name.len() as u32).to_be_bytes());
        pre.extend_from_slice(name.as_bytes());
        pre.extend_from_slice(&(dims.len() as u32).to_be_bytes());
        for d in dims {
            pre.extend_from_slice(&d.to_be_bytes());
        }
    }
    keccak(&[&pre])
}

/// Number of values an adapter contributes.
pub fn value_count(a: &Adapter) -> u64 {
    a.values.iter().map(|v| v.len() as u64).sum()
}

/// `trained − start` on the Q16 grid, canonical order.
pub fn compute(start: &Adapter, trained: &Adapter, scale_log2: u8) -> Result<Vec<i64>, DeltaError> {
    let s = canonical(start);
    let t = canonical(trained);
    if s.len() != t.len() {
        return Err(DeltaError::Mismatch(format!(
            "{} tensors in the start adapter, {} in the trained one",
            s.len(),
            t.len()
        )));
    }
    let scale = (2.0f32).powi(i32::from(scale_log2));
    let mut out = Vec::with_capacity(value_count(start) as usize);
    for ((si, sn, sd), (ti, tn, td)) in s.iter().zip(t.iter()) {
        if sn != tn || sd != td {
            return Err(DeltaError::Mismatch(format!(
                "start has {sn:?} {sd:?} where trained has {tn:?} {td:?}"
            )));
        }
        let sv = &start.values[*si];
        let tv = &trained.values[*ti];
        if sv.len() != tv.len() {
            return Err(DeltaError::Mismatch(format!("{sn:?} value counts differ")));
        }
        for (index, (a, b)) in sv.iter().zip(tv.iter()).enumerate() {
            if !b.is_finite() || !a.is_finite() {
                return Err(DeltaError::NotFinite {
                    tensor: (*sn).to_string(),
                    index,
                });
            }
            let scaled = (b - a) * scale;
            if !scaled.is_finite() || scaled.abs() > MAX_SCALED_ABS {
                return Err(DeltaError::OutOfRange {
                    tensor: (*sn).to_string(),
                    index,
                    value: scaled,
                });
            }
            out.push(Q16::from_f32(scaled).raw());
        }
    }
    Ok(out)
}

/// The start adapter plus an aggregated delta, back in the start adapter's own
/// tensor order. `agg` is Q16 raw, scaled by `2^scale_log2`.
pub fn merge(start: &Adapter, agg: &[i64], scale_log2: u8) -> Result<Vec<Vec<f32>>, DeltaError> {
    if agg.len() as u64 != value_count(start) {
        return Err(DeltaError::Mismatch(format!(
            "aggregate has {} values, the start adapter {}",
            agg.len(),
            value_count(start)
        )));
    }
    let denom = 65536.0f64 * (2.0f64).powi(i32::from(scale_log2));
    let mut merged: Vec<Vec<f32>> = start.values.clone();
    let mut k = 0usize;
    for (fi, _, _) in canonical(start) {
        for v in merged[fi].iter_mut() {
            *v = (f64::from(*v) + agg[k] as f64 / denom) as f32;
            k += 1;
        }
    }
    Ok(merged)
}

/// Serialize the merged adapter as GGUF, copying the start adapter's metadata.
pub fn merged_gguf(start: &Adapter, merged: &[Vec<f32>]) -> Result<Vec<u8>, DeltaError> {
    let tensors: Vec<OutTensor<'_>> = start
        .header
        .tensors
        .iter()
        .zip(merged.iter())
        .map(|(t, v)| OutTensor {
            name: &t.name,
            dims: &t.dims,
            values: v,
        })
        .collect();
    super::gguf::encode_f32(&start.header.kv, &tensors)
        .map_err(|e| DeltaError::Artifact(e.to_string()))
}

/// Chunk `c` of a value vector.
pub fn row(values: &[i64], chunk_dim: u32, c: usize) -> &[i64] {
    let d = chunk_dim as usize;
    let lo = (c * d).min(values.len());
    let hi = (lo + d).min(values.len());
    &values[lo..hi]
}

pub fn chunk_count(n_values: u64, chunk_dim: u32) -> u64 {
    n_values.div_ceil(u64::from(chunk_dim.max(1)))
}

pub fn row_hash(row: &[i64]) -> B32 {
    let mut bytes = Vec::with_capacity(row.len() * 8);
    for v in row {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    keccak(&[&bytes])
}

pub fn row_hashes(values: &[i64], chunk_dim: u32) -> Result<Vec<B32>, DeltaError> {
    if chunk_dim == 0 {
        return Err(DeltaError::ChunkDim);
    }
    let n = chunk_count(values.len() as u64, chunk_dim) as usize;
    Ok((0..n)
        .map(|c| row_hash(row(values, chunk_dim, c)))
        .collect())
}

pub fn delta_root(values: &[i64], chunk_dim: u32) -> Result<B32, DeltaError> {
    let rows = row_hashes(values, chunk_dim)?;
    tree::root(&rows).map_err(|e| DeltaError::Artifact(e.to_string()))
}

/// A delta artifact: what a worker uploads and the coordinator verifies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub value_scale_log2: u8,
    pub round_id: B32,
    pub worker: Addr,
    pub start_adapter_sha256: B32,
    pub trained_adapter_sha256: B32,
    pub manifest_hash: B32,
    pub chunk_dim: u32,
    pub values: Vec<i64>,
}

impl Artifact {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 8 * self.values.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_be_bytes());
        out.push(self.value_scale_log2);
        out.push(0);
        out.extend_from_slice(&self.round_id);
        out.extend_from_slice(&self.worker);
        out.extend_from_slice(&self.start_adapter_sha256);
        out.extend_from_slice(&self.trained_adapter_sha256);
        out.extend_from_slice(&self.manifest_hash);
        out.extend_from_slice(&self.chunk_dim.to_be_bytes());
        out.extend_from_slice(&(self.values.len() as u64).to_be_bytes());
        for v in &self.values {
            out.extend_from_slice(&v.to_be_bytes());
        }
        out
    }

    /// Decode, refusing anything malformed. `max_values` bounds the allocation
    /// before it happens.
    pub fn decode(bytes: &[u8], max_values: u64) -> Result<Self, DeltaError> {
        let bad = |m: &str| DeltaError::Artifact(m.to_string());
        if bytes.len() < HEADER_LEN {
            return Err(bad("shorter than the header"));
        }
        if bytes[0..4] != MAGIC {
            return Err(bad("bad magic"));
        }
        if u16::from_be_bytes([bytes[4], bytes[5]]) != VERSION {
            return Err(bad("unsupported version"));
        }
        if bytes[7] != 0 {
            return Err(bad("reserved byte is not zero"));
        }
        let take32 = |o: usize| {
            let mut a = [0u8; 32];
            a.copy_from_slice(&bytes[o..o + 32]);
            a
        };
        let mut worker = [0u8; 20];
        worker.copy_from_slice(&bytes[40..60]);
        let chunk_dim = u32::from_be_bytes([bytes[156], bytes[157], bytes[158], bytes[159]]);
        let mut n8 = [0u8; 8];
        n8.copy_from_slice(&bytes[160..168]);
        let n = u64::from_be_bytes(n8);
        if n > max_values {
            return Err(DeltaError::TooLarge(n, max_values));
        }
        if chunk_dim == 0 {
            return Err(DeltaError::ChunkDim);
        }
        let body = n
            .checked_mul(8)
            .and_then(|b| usize::try_from(b).ok())
            .ok_or_else(|| bad("value count overflows"))?;
        if bytes.len() != HEADER_LEN + body {
            return Err(bad("length does not match the value count"));
        }
        let values = bytes[HEADER_LEN..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_be_bytes(*c))
            .collect();
        Ok(Self {
            value_scale_log2: bytes[6],
            round_id: take32(8),
            worker,
            start_adapter_sha256: take32(60),
            trained_adapter_sha256: take32(92),
            manifest_hash: take32(124),
            chunk_dim,
            values,
        })
    }
}

#[cfg(test)]
mod tests {
    include!("delta_tests.rs");
}
