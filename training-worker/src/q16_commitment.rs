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
//! **different commitments for the same result**. The tests demonstrate two
//! distinct forms of it, both found by probing the real functions:
//!   1. a one-ULP change in the maximum moves the commitment even when the
//!      quantized payload is byte-identical (the scale is in the preimage);
//!   2. worse, the shared scale couples coordinates — that same one-ULP change
//!      flips a DIFFERENT element's quantized value, so the payload itself is
//!      unstable and no amount of care about hashing would fix it.
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
//! The encoding here does not reimplement that grid — it calls
//! `citrate_fed_types::Q16`, the same kernel `nat-types` re-exports. Pulling it
//! in is why this repo now pins the federation's 1.96.0 toolchain (the kernel's
//! MSRV) instead of floating on "stable".
//!
//! ## Scope
//!
//! This module is the encoding + commitment only. It does not change the default
//! [`crate::backend::ModelBackend::compute_step_commitment`], because that would
//! change what already-deployed workers commit. A backend opts in by overriding
//! that method — which is what the NAT backend does, since its contribution unit
//! is a per-zone Q16 weight delta already.

use citrate_fed_types::Q16;
use sha3::{Digest, Keccak256};

use crate::types::B256;

/// One unit of the Q16.16 grid. Asserted against the real kernel in tests rather
/// than defined independently — see [`to_q16`].
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
/// Delegates to [`citrate_fed_types::Q16::from_f32`] — the SAME function
/// `nat-types` re-exports and the chain's Q16 precompile path uses. This is
/// deliberate: three matching reimplementations of a consensus-critical grid is
/// three chances to drift, and the drift would only surface as honest workers
/// being challenged. Depending on the kernel makes "same grid" structural.
///
/// The properties the federated path relies on are the kernel's, and are pinned
/// by tests here so a kernel bump that changed them fails loudly:
///   - rounds half away from zero, so negation is exact;
///   - non-finite inputs map to ZERO rather than a poison raw;
///   - huge finite inputs saturate rather than wrapping the sign.
pub fn to_q16(values: &[f32]) -> Q16Tensor {
    Q16Tensor {
        values: values.iter().map(|&v| Q16::from_f32(v).raw()).collect(),
    }
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
