//! Commitments on the SHARED Q16 grid — the one NAT and the chain already use.
//!
//! ## The problem this exists to fix
//!
//! [`crate::quantize::quantize_tensor`] encodes a tensor with a **data-dependent
//! float scale**: `scale = max|x| / 127`, values as `i8`. That scale is then
//! hashed *into* the commitment ([`crate::quantize::tensor_commitment`]).
//!
//! So the commitment is a function of `max|x|` as an `f32`. Two workers that
//! compute mathematically identical gradients but accumulate them in a different
//! order — CPU vs GPU, a different batch split, a different reduction tree —
//! can land on `max|x|` values that differ in the last bit, and produce
//! **different commitments for the same result**. `tests` below demonstrates it:
//! a one-ULP change in the maximum element leaves every quantized value
//! bit-identical and still moves the commitment.
//!
//! That is not a theoretical nuisance on this chain. `ComputePoolTraining` has a
//! challenge path with a `CHALLENGE_BOND`, a committee vote, and `SLASH_BPS`
//! (10%) of a worker's stake. A worker slashed for a last-bit difference in a
//! float reduction was not dishonest; it was on different hardware. Federated
//! training is heterogeneous hardware by definition.
//!
//! ## Why Q16
//!
//! NAT already settled this. `nat_types::Q16` is re-exported from
//! `citrate-fed-types` precisely "so NAT and the chain run the *same* Q16 grid by
//! construction", and `nat-federated`'s seam requires aggregation to "ride the
//! Q16 precompile path (`0x0110` family), never the f32 reference impl … that is
//! the only path that is bit-reproducible across heterogeneous validators"
//! (ADR-0006 / `MergeDeterminism.tla`).
//!
//! Q16 is a **fixed** grid: one unit is `1/65536`, independent of the data. Two
//! nodes that agree on the value agree on the bits, on any hardware, with no
//! shared scale to negotiate.
//!
//! ## Scope
//!
//! This module is the encoding + commitment only. It does not change the default
//! [`crate::backend::ModelBackend::compute_step_commitment`], because that would
//! change what already-deployed workers commit. A backend opts in by overriding
//! that method — which is what the NAT backend does, since its contribution unit
//! is a per-zone Q16 weight delta already.

use sha3::{Digest, Keccak256};

use crate::types::B256;

/// One unit of the Q16.16 grid, matching `citrate_fed_types::Q16`
/// (`FRAC_BITS = 16`, so `ONE_RAW = 65536`).
pub const Q16_ONE: i64 = 1 << 16;

/// A tensor encoded on the fixed Q16 grid.
///
/// Deliberately carries **no scale**: the grid is the same everywhere, which is
/// the entire point. There is nothing hardware-dependent to hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q16Tensor {
    pub values: Vec<i64>,
}

/// Encode f32s onto the fixed Q16 grid.
///
/// Rounds half-away-from-zero, which is symmetric about zero — so negating a
/// gradient negates its encoding exactly, and a sign convention cannot introduce
/// a one-unit asymmetry between two workers.
///
/// Saturates rather than wrapping: a gradient beyond the Q16 range is clamped to
/// the representable extreme, never wrapped into a value of the opposite sign.
/// Non-finite inputs encode as 0 — a NaN gradient is a bug upstream, and it must
/// not become an unpredictable commitment.
pub fn to_q16(values: &[f32]) -> Q16Tensor {
    let out = values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return 0i64;
            }
            let scaled = (v as f64) * (Q16_ONE as f64);
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5).floor()
            } else {
                (scaled - 0.5).ceil()
            };
            if rounded >= i64::MAX as f64 {
                i64::MAX
            } else if rounded <= i64::MIN as f64 {
                i64::MIN
            } else {
                rounded as i64
            }
        })
        .collect();
    Q16Tensor { values: out }
}

/// Per-tensor commitment on the Q16 grid: `keccak256(be_bytes(v) for v in values)`.
///
/// Big-endian so the digest does not depend on host endianness, and no scale in
/// the preimage because there is no scale.
pub fn q16_tensor_commitment(t: &Q16Tensor) -> B256 {
    let mut h = Keccak256::new();
    for v in &t.values {
        h.update(v.to_be_bytes());
    }
    let out = h.finalize();
    let mut b = [0u8; 32];
    b.copy_from_slice(&out);
    B256::from(b)
}

/// Step commitment over several tensors in canonical layer order.
///
/// The layer index is folded in explicitly, so two tensors that happen to hold
/// the same numbers in different layers cannot produce the same preimage — and
/// reordering the layers changes the commitment rather than silently passing.
pub fn q16_step_commitment(tensors: &[(u32, Q16Tensor)]) -> B256 {
    let mut ordered: Vec<&(u32, Q16Tensor)> = tensors.iter().collect();
    ordered.sort_by_key(|(idx, _)| *idx);

    let mut h = Keccak256::new();
    for (idx, t) in ordered {
        h.update(idx.to_be_bytes());
        h.update(q16_tensor_commitment(t).as_bytes());
    }
    let out = h.finalize();
    let mut b = [0u8; 32];
    b.copy_from_slice(&out);
    B256::from(b)
}

#[cfg(test)]
mod tests {
    include!("q16_commitment_tests.rs");
}
