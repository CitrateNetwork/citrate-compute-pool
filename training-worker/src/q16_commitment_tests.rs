// Q16 commitment tests.
//
// The first two are the load-bearing ones. They demonstrate, on the EXISTING
// f32-scale path, TWO distinct failures under a one-ULP perturbation of a single
// element — both found empirically, not assumed:
//
//   1. the commitment moves because the derived f32 scale is in the preimage,
//      even when the quantized payload is byte-identical;
//   2. worse, the shared scale COUPLES coordinates: perturbing element 0 flips
//      element 1's quantized value, though element 1 never changed.
//
// (2) is why this cannot be fixed by hashing more carefully. The payload itself
// is unstable under last-bit noise. A fixed grid makes coordinates independent.

use super::*;
use crate::quantize::{quantize_tensor, tensor_commitment};

// ── The hazard, demonstrated on the existing path ────────────────────────

/// THE MOTIVATING DEFECT, form 1: the SCALE alone moves the commitment.
///
/// `quantize_tensor` derives `scale = max|x| / 127`, and `tensor_commitment`
/// hashes that f32 scale into the preimage. For `[1.0, 0.25]` a one-ULP bump to
/// the maximum leaves the quantized payload byte-for-byte identical and still
/// produces a different commitment.
///
/// Two honest workers on different hardware reach `max|x|` by different reduction
/// orders. This is how one gets challenged and slashed 10% for arithmetic it did
/// not get wrong.
#[test]
fn the_f32_scale_alone_changes_the_commitment_on_a_one_ulp_perturbation() {
    let a: Vec<f32> = vec![1.0, 0.25];
    let mut b = a.clone();
    b[0] = f32::from_bits(a[0].to_bits() + 1);

    let qa = quantize_tensor(&a);
    let qb = quantize_tensor(&b);

    assert_eq!(
        qa.q_values, qb.q_values,
        "the QUANTIZED PAYLOAD is identical here — the entire difference is the \
         scale, which is why this is so easy to miss"
    );
    assert_ne!(qa.scale, qb.scale, "the data-dependent scale moved");
    assert_ne!(
        tensor_commitment(&qa),
        tensor_commitment(&qb),
        "THE DEFECT: identical quantized values, different commitment, because the \
         f32 scale is in the preimage"
    );
}

/// THE MOTIVATING DEFECT, form 2 — the worse one: the shared scale COUPLES EVERY
/// COORDINATE TO THE MAXIMUM.
///
/// Because every value is divided by the same derived scale, a one-ULP change in
/// the maximum element pushes OTHER coordinates across their rounding boundary.
/// Here a last-bit change to element 0 flips element 1 from 64 to 63 — element 1
/// itself never changed.
///
/// So this is not merely "the scale is in the hash". The quantized payload is
/// itself unstable under last-bit noise in a single coordinate, which no amount
/// of care about hashing would fix. On a fixed grid the coordinates are
/// independent and this cannot happen.
#[test]
fn the_f32_scale_couples_every_coordinate_to_the_maximum() {
    let a: Vec<f32> = vec![1.0, 0.5, -0.25];
    let mut b = a.clone();
    b[0] = f32::from_bits(a[0].to_bits() + 1);

    let qa = quantize_tensor(&a);
    let qb = quantize_tensor(&b);

    assert_eq!(qa.q_values[0], qb.q_values[0], "the max pins to 127 either way");
    assert_ne!(
        qa.q_values[1], qb.q_values[1],
        "element 1 was NOT touched, yet its quantized value moved ({} -> {}) \
         because it shares a scale derived from element 0",
        qa.q_values[1], qb.q_values[1]
    );

    // And the same coupling on the Q16 grid: nothing moves.
    assert_eq!(
        to_q16(&a).values[1],
        to_q16(&b).values[1],
        "on a fixed grid, coordinates are independent — perturbing element 0 \
         cannot move element 1"
    );
}

/// The same perturbation on the Q16 grid is inert: there is no scale to move,
/// and one ULP of an f32 near 1.0 is ~6e-8, far below the 1/65536 grid step.
#[test]
fn the_q16_path_is_immune_to_the_same_perturbation() {
    let a: Vec<f32> = vec![1.0, 0.5, -0.25];
    let mut b = a.clone();
    b[0] = f32::from_bits(a[0].to_bits() + 1);

    assert_eq!(to_q16(&a), to_q16(&b), "same grid point");
    assert_eq!(
        q16_tensor_commitment(&to_q16(&a)),
        q16_tensor_commitment(&to_q16(&b)),
        "the whole point: identical results commit identically, on any hardware"
    );
}

/// A change LARGER than the grid step must still be caught — immunity to noise
/// must not become blindness to real divergence, or the challenge path is
/// worthless.
#[test]
fn a_real_difference_still_changes_the_q16_commitment() {
    let a: Vec<f32> = vec![1.0, 0.5];
    let b: Vec<f32> = vec![1.0, 0.5 + 1.0 / 65536.0];
    assert_ne!(
        q16_tensor_commitment(&to_q16(&a)),
        q16_tensor_commitment(&to_q16(&b)),
        "one full grid step apart must be distinguishable"
    );
}

// ── Grid properties the federated path depends on ────────────────────────

/// The grid matches `citrate_fed_types::Q16` exactly: one unit is 1/65536.
/// If this drifts, NAT and the chain stop agreeing and the whole "same grid by
/// construction" guarantee is gone.
#[test]
fn the_grid_matches_the_shared_fed_types_grid() {
    assert_eq!(Q16_ONE, 65536, "FRAC_BITS = 16");
    assert_eq!(to_q16(&[1.0]).values, vec![65536]);
    assert_eq!(to_q16(&[0.5]).values, vec![32768]);
    assert_eq!(to_q16(&[-1.0]).values, vec![-65536]);
    assert_eq!(to_q16(&[0.0]).values, vec![0]);
}

/// Rounding is symmetric about zero, so negating a gradient negates its encoding
/// exactly. An asymmetric rule (round-half-up, say) would make `-x` encode one
/// unit away from `-(x)`, and two workers with opposite sign conventions would
/// disagree by one unit on half their coordinates.
#[test]
fn rounding_is_symmetric_about_zero() {
    for v in [0.3f32, 1.7, 0.000_01, 12.5, 1e-5] {
        let pos = to_q16(&[v]).values[0];
        let neg = to_q16(&[-v]).values[0];
        assert_eq!(pos, -neg, "encoding of {v} and {} must be exact negatives", -v);
    }
}

/// Non-finite gradients encode as 0 rather than an unpredictable bit pattern. A
/// NaN is a bug upstream; it must not become a commitment nobody can reproduce.
#[test]
fn non_finite_values_encode_as_zero_not_garbage() {
    assert_eq!(to_q16(&[f32::NAN]).values, vec![0]);
    assert_eq!(to_q16(&[f32::INFINITY]).values, vec![0]);
    assert_eq!(to_q16(&[f32::NEG_INFINITY]).values, vec![0]);
}

/// Out-of-range values saturate rather than wrapping. Wrapping would turn a huge
/// positive gradient into a large negative one — a sign flip in a settlement
/// input, which is the worst possible failure mode here.
#[test]
fn extreme_values_saturate_rather_than_wrapping() {
    let huge = to_q16(&[f32::MAX]).values[0];
    let tiny = to_q16(&[f32::MIN]).values[0];
    assert!(huge > 0, "a huge positive must stay positive, got {huge}");
    assert!(tiny < 0, "a huge negative must stay negative, got {tiny}");
}

// ── Step commitment ──────────────────────────────────────────────────────

/// Layer order is part of the commitment, and supplying layers out of order does
/// not change it — the tensors are sorted by index first. A worker that emits its
/// layers in a different order must still commit the same root.
#[test]
fn step_commitment_is_layer_ordered_not_submission_ordered() {
    let l0 = to_q16(&[1.0, 2.0]);
    let l1 = to_q16(&[3.0, 4.0]);

    let in_order = q16_step_commitment(&[(0, l0.clone()), (1, l1.clone())]);
    let shuffled = q16_step_commitment(&[(1, l1), (0, l0)]);
    assert_eq!(in_order, shuffled, "submission order must not matter");
}

/// The same numbers in DIFFERENT layers must not collide. Folding the layer index
/// into the preimage is what prevents a worker swapping two layers' gradients and
/// still committing a valid-looking root.
#[test]
fn the_same_values_in_different_layers_do_not_collide() {
    let t = to_q16(&[1.0, 2.0]);
    let a = q16_step_commitment(&[(0, t.clone())]);
    let b = q16_step_commitment(&[(1, t)]);
    assert_ne!(a, b, "the layer index must be bound into the commitment");
}

/// Determinism across repeated calls — the floor everything else stands on.
#[test]
fn the_commitment_is_deterministic() {
    let t = to_q16(&[0.1, -0.2, 0.3]);
    assert_eq!(q16_tensor_commitment(&t), q16_tensor_commitment(&t));
    let s = vec![(0u32, t)];
    assert_eq!(q16_step_commitment(&s), q16_step_commitment(&s));
}
