//! Attributing a checkpoint delta to NAT zones — the link between this worker's
//! flat gradient tensors and the co-op's per-zone contribution unit.
//!
//! ## The mismatch this closes
//!
//! `ModelBackend::forward_backward` returns flat
//! [`crate::backend::Tensor`]s carrying a `layer_index`. NAT's federated seam
//! aggregates something different: `ZoneWeightDelta { zone, delta }` — a delta
//! **per zone**, Belnap-aggregated coordinate-wise on the Q16 precompile path.
//! Without a mapping between the two, a compute-pool worker cannot contribute to
//! a NAT federated round at all, whatever else it computes correctly.
//!
//! ## Why no change to NAT is needed
//!
//! `nat_candle::autoreg::AutoregLm` registers its parameters with names that
//! already carry zone identity:
//!
//! ```text
//!   zone_HP.wq  zone_HP.wk  zone_HP.wv  zone_HP.wo     attention cores (HP/PF/CX)
//!   zone_SM.wb  zone_SM.wc  zone_SM.wo  zone_SM.log_a  SSM cores (SM/CB)
//!   score_HP                                            per-zone merge score head
//!   embedding.weight  readout                           shared, not zone-owned
//! ```
//!
//! So a checkpoint saved before and after a step can be differenced tensor by
//! tensor and the result grouped by that prefix. The attribution is read off the
//! model's own naming rather than inferred from shapes or positions, which is
//! what makes it safe: a renamed parameter fails to attribute loudly instead of
//! landing in the wrong zone's delta.
//!
//! ## What is deliberately NOT attributed
//!
//! `embedding.weight` and `readout` are shared across zones. They are real
//! trained parameters, but they belong to no zone, so folding them into one
//! would misreport that zone's contribution. They are reported separately as
//! [`SHARED`] and it is the caller's decision what to do with them — silently
//! attributing them somewhere is exactly the kind of quiet wrongness that is
//! invisible once it reaches a settlement ledger.
//!
//! `MX` never appears: it is the non-learned executive harness and has no
//! parameters. `nat_federated::ZoneWeightDelta::new` rejects it outright
//! (`SeamError::NotALearnedZone`), so producing an MX delta here would be
//! constructing something the seam is guaranteed to refuse.

use std::collections::BTreeMap;

/// The bucket for parameters that are trained but belong to no single zone.
pub const SHARED: &str = "__shared__";

/// Prefix marking a zone-owned core parameter: `zone_HP.wq`.
const ZONE_PREFIX: &str = "zone_";
/// Prefix marking a zone's merge score head: `score_HP`.
const SCORE_PREFIX: &str = "score_";

/// Which zone a checkpoint tensor belongs to, by NAT's own naming.
///
/// Returns [`SHARED`] for trained-but-unowned parameters. `None` is not used:
/// every parameter lands somewhere explicit, so a tensor cannot be dropped from
/// the accounting by accident.
pub fn zone_of(tensor_name: &str) -> &str {
    if let Some(rest) = tensor_name.strip_prefix(ZONE_PREFIX) {
        // `zone_HP.wq` -> `HP`; a bare `zone_HP` (no field) is still that zone.
        return rest.split('.').next().unwrap_or(SHARED);
    }
    if let Some(rest) = tensor_name.strip_prefix(SCORE_PREFIX) {
        // `score_HP` -> `HP`. The merge score head is that zone's parameter.
        return rest.split('.').next().unwrap_or(SHARED);
    }
    SHARED
}

/// A per-zone delta, ready to become a `nat_federated::ZoneWeightDelta`.
///
/// `values` are ordered by tensor name so two workers that iterate a checkpoint
/// in different order still produce the same vector — the aggregation is
/// coordinate-wise, so a permutation would silently corrupt it rather than fail.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneDelta {
    pub zone: String,
    pub values: Vec<f32>,
}

/// Difference two checkpoints and group the result by zone.
///
/// Both sides are `(tensor_name, values)`. Ordering of the inputs does not
/// matter; the output is deterministic — zones sorted, and within a zone the
/// tensors concatenated in name order.
///
/// A tensor present on one side and not the other, or present with a different
/// length, is a [`DeltaError`]: it means the two checkpoints are not the same
/// model, and differencing them would produce a number that looks like a
/// gradient and is not one.
pub fn zone_deltas(
    before: &[(String, Vec<f32>)],
    after: &[(String, Vec<f32>)],
) -> Result<Vec<ZoneDelta>, DeltaError> {
    let pre: BTreeMap<&str, &Vec<f32>> = before.iter().map(|(n, v)| (n.as_str(), v)).collect();
    let post: BTreeMap<&str, &Vec<f32>> = after.iter().map(|(n, v)| (n.as_str(), v)).collect();

    if pre.len() != before.len() || post.len() != after.len() {
        return Err(DeltaError::DuplicateTensor);
    }
    for name in pre.keys() {
        if !post.contains_key(name) {
            return Err(DeltaError::TensorMissing((*name).to_string()));
        }
    }
    for name in post.keys() {
        if !pre.contains_key(name) {
            return Err(DeltaError::TensorMissing((*name).to_string()));
        }
    }

    // BTreeMap iteration is name-ordered, so the concatenation is deterministic.
    let mut by_zone: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (name, a) in &pre {
        let b = post[name];
        if a.len() != b.len() {
            return Err(DeltaError::ShapeMismatch {
                tensor: (*name).to_string(),
                before: a.len(),
                after: b.len(),
            });
        }
        let bucket = by_zone.entry(zone_of(name).to_string()).or_default();
        bucket.extend(b.iter().zip(a.iter()).map(|(x, y)| x - y));
    }

    Ok(by_zone
        .into_iter()
        .map(|(zone, values)| ZoneDelta { zone, values })
        .collect())
}

/// Why a checkpoint pair could not be differenced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaError {
    /// A tensor exists on one side only — the checkpoints are different models.
    TensorMissing(String),
    /// A tensor changed length between checkpoints.
    ShapeMismatch {
        tensor: String,
        before: usize,
        after: usize,
    },
    /// The same tensor name appeared twice, so the pairing is ambiguous.
    DuplicateTensor,
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::TensorMissing(t) => write!(
                f,
                "tensor '{t}' is present in only one checkpoint — these are not the \
                 same model, and differencing them would yield a number that looks \
                 like a gradient without being one"
            ),
            DeltaError::ShapeMismatch {
                tensor,
                before,
                after,
            } => write!(
                f,
                "tensor '{tensor}' changed length {before} -> {after} between \
                 checkpoints"
            ),
            DeltaError::DuplicateTensor => {
                write!(f, "a tensor name appeared twice; the pairing is ambiguous")
            }
        }
    }
}

impl std::error::Error for DeltaError {}

#[cfg(test)]
mod tests {
    include!("zone_delta_tests.rs");
}
