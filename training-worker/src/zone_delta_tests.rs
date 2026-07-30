// Zone-attribution tests.
//
// The names below are the REAL ones `nat_candle::autoreg::AutoregLm` registers
// (`zone_HP.wq`, `score_PF`, `embedding.weight`, `readout`). Attribution is read
// off the model's own naming, so these fixtures are the contract: if NAT renames
// a parameter, these fail rather than the delta quietly landing in the wrong
// zone — which would misreport a member's contribution to a settlement ledger.

use super::*;

fn ck(pairs: &[(&str, &[f32])]) -> Vec<(String, Vec<f32>)> {
    pairs
        .iter()
        .map(|(n, v)| ((*n).to_string(), v.to_vec()))
        .collect()
}

fn delta_for<'a>(ds: &'a [ZoneDelta], zone: &str) -> &'a ZoneDelta {
    ds.iter()
        .find(|d| d.zone == zone)
        .unwrap_or_else(|| panic!("no delta for zone {zone}; got {:?}", ds.iter().map(|d| &d.zone).collect::<Vec<_>>()))
}

// ── Attribution by NAT's real naming ─────────────────────────────────────

/// Attention-core parameters attribute to their zone. These are the exact names
/// `AutoregLm` builds for HP/PF/CX.
#[test]
fn attention_core_parameters_attribute_to_their_zone() {
    assert_eq!(zone_of("zone_HP.wq"), "HP");
    assert_eq!(zone_of("zone_PF.wk"), "PF");
    assert_eq!(zone_of("zone_CX.wv"), "CX");
    assert_eq!(zone_of("zone_CX.wo"), "CX");
}

/// SSM-core parameters likewise, including the scalar `log_a` that only SSM
/// zones have.
#[test]
fn ssm_core_parameters_attribute_to_their_zone() {
    assert_eq!(zone_of("zone_SM.wb"), "SM");
    assert_eq!(zone_of("zone_CB.wc"), "CB");
    assert_eq!(zone_of("zone_SM.log_a"), "SM");
}

/// The per-zone merge score head belongs to that zone. It is named
/// `score_HP`, not `zone_HP.score`, so it needs its own rule — and missing it
/// would silently dump every zone's score head into the shared bucket.
#[test]
fn the_merge_score_head_belongs_to_its_zone() {
    assert_eq!(zone_of("score_HP"), "HP");
    assert_eq!(zone_of("score_SM"), "SM");
}

/// Embedding and readout are trained but zone-less. They must land in SHARED,
/// not in whichever zone happens to sort first — attributing them to a zone
/// would inflate that zone's reported contribution.
#[test]
fn shared_parameters_are_not_attributed_to_any_zone() {
    assert_eq!(zone_of("embedding.weight"), SHARED);
    assert_eq!(zone_of("readout"), SHARED);
    assert_eq!(zone_of("readout.bias"), SHARED);
}

/// An unrecognised name lands in SHARED rather than being dropped. Dropping it
/// would remove real trained parameters from the accounting silently; SHARED at
/// least keeps it visible and unattributed.
#[test]
fn an_unrecognised_parameter_is_shared_not_discarded() {
    assert_eq!(zone_of("some_future_param"), SHARED);
}

// ── Differencing ─────────────────────────────────────────────────────────

/// The end-to-end shape the co-op needs: a step's checkpoint delta, grouped per
/// zone, ready to become `nat_federated::ZoneWeightDelta`.
#[test]
fn a_checkpoint_delta_groups_by_zone() {
    let before = ck(&[
        ("zone_HP.wq", &[1.0, 2.0]),
        ("zone_PF.wq", &[10.0]),
        ("embedding.weight", &[100.0]),
    ]);
    let after = ck(&[
        ("zone_HP.wq", &[1.5, 2.25]),
        ("zone_PF.wq", &[9.0]),
        ("embedding.weight", &[100.5]),
    ]);

    let ds = zone_deltas(&before, &after).expect("same model, must difference");
    assert_eq!(ds.len(), 3, "HP, PF, and SHARED");
    assert_eq!(delta_for(&ds, "HP").values, vec![0.5, 0.25]);
    assert_eq!(delta_for(&ds, "PF").values, vec![-1.0], "deltas are signed");
    assert_eq!(delta_for(&ds, SHARED).values, vec![0.5]);
}

/// Several tensors in one zone concatenate in NAME order, not input order. The
/// aggregation is coordinate-wise across workers, so two workers that enumerated
/// their checkpoint differently must still produce the same vector — otherwise
/// the aggregate is silently mixing unrelated coordinates.
#[test]
fn tensors_within_a_zone_concatenate_in_name_order_regardless_of_input_order() {
    let before = ck(&[("zone_HP.wq", &[0.0]), ("zone_HP.wk", &[0.0])]);
    let after = ck(&[("zone_HP.wq", &[1.0]), ("zone_HP.wk", &[2.0])]);

    // Same content, reversed enumeration.
    let before_rev = ck(&[("zone_HP.wk", &[0.0]), ("zone_HP.wq", &[0.0])]);
    let after_rev = ck(&[("zone_HP.wk", &[2.0]), ("zone_HP.wq", &[1.0])]);

    let a = zone_deltas(&before, &after).expect("a");
    let b = zone_deltas(&before_rev, &after_rev).expect("b");
    assert_eq!(a, b, "enumeration order must not change the delta vector");
    // wk sorts before wq.
    assert_eq!(delta_for(&a, "HP").values, vec![2.0, 1.0]);
}

/// `MX` has no parameters, so it never appears. Emitting an MX delta would build
/// something `ZoneWeightDelta::new` is guaranteed to reject
/// (`SeamError::NotALearnedZone`) — better never to construct it.
#[test]
fn the_non_learned_mx_harness_never_produces_a_delta() {
    let before = ck(&[("zone_HP.wq", &[0.0]), ("embedding.weight", &[0.0])]);
    let after = ck(&[("zone_HP.wq", &[1.0]), ("embedding.weight", &[1.0])]);
    let ds = zone_deltas(&before, &after).expect("difference");
    assert!(
        ds.iter().all(|d| d.zone != "MX"),
        "MX is the non-learned harness and owns no parameters"
    );
}

/// A zone that did not move still reports — a zero delta is a real, meaningful
/// contribution record. Omitting it would let a worker's absence look the same
/// as a worker that trained and produced no change.
#[test]
fn a_zone_that_did_not_change_reports_a_zero_delta() {
    let before = ck(&[("zone_HP.wq", &[1.0, 2.0])]);
    let after = ck(&[("zone_HP.wq", &[1.0, 2.0])]);
    let ds = zone_deltas(&before, &after).expect("difference");
    assert_eq!(delta_for(&ds, "HP").values, vec![0.0, 0.0]);
}

// ── Refusals ─────────────────────────────────────────────────────────────

/// Checkpoints from DIFFERENT models must not be differenced. The result would
/// be a number shaped exactly like a gradient that is not one, and it would flow
/// straight into a settlement ledger.
#[test]
fn checkpoints_of_different_models_are_refused() {
    let before = ck(&[("zone_HP.wq", &[1.0])]);
    let after = ck(&[("zone_HP.wq", &[1.0]), ("zone_PF.wq", &[2.0])]);
    let err = zone_deltas(&before, &after).expect_err("different tensor sets");
    assert_eq!(err, DeltaError::TensorMissing("zone_PF.wq".into()));
    assert!(
        err.to_string().contains("not the same model"),
        "the message must say why this is refused: {err}"
    );
}

/// A tensor that changed length between checkpoints is the same problem in a
/// subtler form — same names, different architecture.
#[test]
fn a_tensor_that_changed_shape_is_refused() {
    let before = ck(&[("zone_HP.wq", &[1.0, 2.0])]);
    let after = ck(&[("zone_HP.wq", &[1.0])]);
    let err = zone_deltas(&before, &after).expect_err("shape changed");
    assert!(matches!(err, DeltaError::ShapeMismatch { .. }), "{err:?}");
}

/// A duplicated tensor name makes the pairing ambiguous — refuse rather than
/// silently taking whichever one happened to be last.
#[test]
fn a_duplicated_tensor_name_is_refused() {
    let dup = ck(&[("zone_HP.wq", &[1.0]), ("zone_HP.wq", &[2.0])]);
    let ok = ck(&[("zone_HP.wq", &[1.0])]);
    assert_eq!(
        zone_deltas(&dup, &ok).expect_err("ambiguous"),
        DeltaError::DuplicateTensor
    );
}
