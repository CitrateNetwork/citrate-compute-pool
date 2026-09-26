// Registration is the only place an untrusted party tells the coordinator
// something about itself. Everything downstream — which jobs it sees, whether its
// results count — follows from what is accepted here.

use super::*;
use citrate_training_worker::coordinator_protocol::attestation_digest;
use citrate_training_worker::wallet::Wallet;

// Anvil account #0 / #1. Well-known throwaways, used so the derived addresses are
// deterministic across runs. NOT real keys.
const KEY_A: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const KEY_B: &str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";

/// A probe document shaped exactly like the CUDA run measured on the GB10.
fn probe_json(backend: &str, dtype: &str, tok_s: f64, self_repeat: bool) -> String {
    serde_json::json!({
        "schema": PROBE_SCHEMA,
        "backend": backend,
        "dtype": dtype,
        "os": "linux",
        "arch": "aarch64",
        "perf": { "tokens_per_second": tok_s },
        "self_repeat_identical": self_repeat,
    })
    .to_string()
}

/// Signs through the SHARED digest, which is what a real worker calls. If this
/// crate ever grew its own copy, this helper would keep passing while the fleet
/// failed — so the test signs the way the wire does.
fn sign(key: &str, body: &str) -> Attestation {
    let w = Wallet::from_hex(key).expect("load test key");
    Attestation {
        probe_json: body.to_string(),
        timestamp: 1,
        signature: w
            .sign_digest_recoverable(&attestation_digest(body, 1))
            .expect("sign")
            .to_vec(),
    }
}

// ── Capability derivation ──────────────────────────────────────────────

fn cap(backend: &str, dtype: &str, tok_s: f64, self_repeat: bool) -> Capability {
    let p: ProbeReport =
        serde_json::from_str(&probe_json(backend, dtype, tok_s, self_repeat)).unwrap();
    capability_of(&p)
}

/// f32 on any self-repeating accelerator earns the ladder. The dtype moved from
/// bf16 because bf16 is not run-to-run deterministic on Metal, so a bf16 ladder
/// would be one CUDA machine's ladder by construction.
#[test]
fn f32_on_a_self_repeating_accelerator_earns_the_ladder() {
    assert_eq!(cap("candle-cuda", "f32", 71_098.0, true), Capability::H01);
    assert_eq!(cap("candle-metal", "f32", 13_472.0, true), Capability::H01);
}

/// The M2 Max that actually reported in: 13,472 tok/s on Metal f32. The original
/// 15,000 floor excluded it, which was an artifact of interpolating from one
/// machine rather than a judgement about the hardware.
#[test]
fn the_first_real_apple_machine_clears_the_floor() {
    const { assert!(13_472.0 >= ACCELERATOR_TOK_S) };
    assert_eq!(cap("candle-metal", "f32", 13_472.0, true), Capability::H01);
}

/// And every measured CPU stays below it.
#[test]
fn measured_cpus_remain_below_the_accelerator_floor() {
    const { assert!(7_786.0 < ACCELERATOR_TOK_S, "GB10 CPU") };
    const { assert!(7_137.0 < ACCELERATOR_TOK_S, "M2 Max CPU") };
}

/// The load-bearing gate. A device that cannot reproduce its own result cannot
/// have its work verified by recomputation, so it must never receive work whose
/// settlement depends on that — however fast it is.
#[test]
fn a_device_that_fails_self_repeat_is_capped_at_probe_however_fast() {
    assert_eq!(
        cap("candle-cuda", "f32", 1_000_000.0, false),
        Capability::Probe
    );
}

/// bf16 can co-train but must not get ladder work: unverifiable on Metal
/// (3/3 runs differ) and unmeasured for cross-backend divergence everywhere.
#[test]
fn bf16_earns_co_training_but_not_the_ladder() {
    assert_eq!(
        cap("candle-cuda", "bf16", 71_098.0, true),
        Capability::Federated
    );
    assert_eq!(
        cap("candle-metal", "bf16", 13_649.0, true),
        Capability::Federated
    );
}

#[test]
fn apple_silicon_f32_now_earns_the_ladder() {
    assert_eq!(cap("candle-metal", "f32", 40_000.0, true), Capability::H01);
}

#[test]
fn cpu_maps_divergence() {
    // 7,786 tok/s is the measured GB10 CPU figure.
    assert_eq!(cap("candle-cpu", "f32", 7_786.0, true), Capability::Probe);
}

#[test]
fn an_accelerator_below_the_throughput_floor_is_treated_as_cpu_class() {
    assert_eq!(
        cap("candle-cuda", "f32", ACCELERATOR_TOK_S - 1.0, true),
        Capability::Probe
    );
}

/// CP-B-001: `tokens_per_second` is attacker-written. A claim of impossible
/// throughput on the fixed probe job must not mint the top `H01` tier — it is a
/// fabrication, not a measurement, and is treated as unverified.
#[test]
fn an_impossible_throughput_claim_does_not_mint_the_top_tier() {
    assert_eq!(
        cap("candle-cuda", "f32", 999_999.0, true),
        Capability::Probe
    );
    // The real fleet's fastest measurement stays a valid ladder machine.
    assert_eq!(cap("candle-cuda", "f32", 71_098.0, true), Capability::H01);
}

#[test]
fn an_unknown_backend_is_not_trusted_with_more_than_probe() {
    assert_eq!(
        cap("candle-something-new", "f32", 90_000.0, true),
        Capability::Probe
    );
}

// ── Verification ───────────────────────────────────────────────────────

#[test]
fn a_signed_probe_registers_the_signing_address_as_the_worker() {
    let body = probe_json("candle-cuda", "f32", 71_098.0, true);
    let w = verify(&sign(KEY_A, &body)).expect("verify");
    assert_eq!(w.id, Wallet::from_hex(KEY_A).unwrap().address());
    assert_eq!(w.capability, Capability::H01);
}

/// The property that removes the roster: two different keys register as two
/// different workers with no registry consulted.
#[test]
fn two_keys_register_as_two_distinct_workers() {
    let body = probe_json("candle-cuda", "bf16", 71_098.0, true);
    let a = verify(&sign(KEY_A, &body)).unwrap();
    let b = verify(&sign(KEY_B, &body)).unwrap();
    assert_ne!(a.id, b.id);
}

/// The reason the probe is signed at all — and the exact shape of the guarantee,
/// which is easy to state wrongly.
///
/// secp256k1 recovery does NOT fail on a tampered message: it succeeds and yields
/// a DIFFERENT address. So editing the probe does not produce a signature error,
/// it produces a registration for an identity whose key the tamperer does not
/// hold. The property is therefore not "tampering is rejected" but the stronger
/// and more useful **"a key cannot be made to vouch for a machine it did not
/// measure"**. See `a_tampered_registration_is_unusable` for why the resulting
/// junk identity buys nothing.
#[test]
fn a_key_cannot_be_made_to_vouch_for_a_better_machine() {
    let honest = probe_json("candle-cpu", "f32", 7_786.0, true);
    let mut att = sign(KEY_A, &honest);
    let real = verify(&att).unwrap();
    assert_eq!(real.capability, Capability::Probe);
    assert_eq!(real.id, Wallet::from_hex(KEY_A).unwrap().address());

    att.probe_json = probe_json("candle-cuda", "bf16", 71_098.0, true);
    let forged = verify(&att).expect("recovery still succeeds — that is the point");
    assert_ne!(
        forged.id, real.id,
        "the upgraded claim must not register under the honest signer's identity"
    );
    assert_ne!(forged.id, Wallet::from_hex(KEY_B).unwrap().address());
}

/// Specifically the self-repeat flag, since flipping one boolean is the cheapest
/// possible lie and the one that would do the most damage.
#[test]
fn flipping_the_self_repeat_flag_changes_the_identity_it_registers() {
    let honest = probe_json("candle-cuda", "bf16", 71_098.0, false);
    let mut att = sign(KEY_A, &honest);
    let real = verify(&att).unwrap();
    assert_eq!(
        real.capability,
        Capability::Probe,
        "self-repeat false caps at probe"
    );

    att.probe_json = probe_json("candle-cuda", "bf16", 71_098.0, true);
    assert_ne!(verify(&att).unwrap().id, real.id);
}

#[test]
fn a_document_from_another_schema_is_refused() {
    let att = sign(
        KEY_A,
        &serde_json::json!({ "schema": "some/other" }).to_string(),
    );
    assert!(matches!(verify(&att), Err(AttestError::WrongSchema(_))));
}

/// Registration is unauthenticated network input. Every malformed shape must be
/// an ordinary error — a panic here is a denial of service on the coordinator by
/// anyone who can reach it.
#[test]
fn malformed_attestations_are_rejected_without_panicking() {
    let body = probe_json("candle-cuda", "bf16", 71_098.0, true);
    let good = sign(KEY_A, &body);

    for (label, att) in [
        (
            "not json",
            Attestation { timestamp: 1, probe_json: "{oops".into(), signature: good.signature.clone() },
        ),
        (
            "empty signature",
            Attestation { timestamp: 1, probe_json: body.clone(), signature: vec![] },
        ),
        (
            "short signature",
            Attestation { timestamp: 1, probe_json: body.clone(), signature: vec![0u8; 64] },
        ),
        (
            "all-zero signature",
            Attestation { timestamp: 1, probe_json: body.clone(), signature: vec![0u8; 65] },
        ),
        (
            "bad recovery id",
            Attestation {
                probe_json: body.clone(),
                timestamp: 1,
                signature: {
                    let mut v = good.signature.clone();
                    v[64] = 99;
                    v
                },
            },
        ),
        (
            "missing perf",
            sign(KEY_A, &serde_json::json!({ "schema": PROBE_SCHEMA, "backend": "candle-cpu", "dtype": "f32", "self_repeat_identical": true }).to_string()),
        ),
    ] {
        assert!(verify(&att).is_err(), "{label} must be rejected");
    }
}

/// The probe bytes are stored verbatim rather than re-serialized, because JSON
/// does not round-trip byte-for-byte (key order, number formatting) and a
/// signature over re-encoded bytes would fail for honest workers.
#[test]
fn verification_uses_the_bytes_as_signed_not_a_reencoding() {
    // Same document, unusual but legal whitespace and key order.
    let body = "{ \"self_repeat_identical\" : true ,\n \"schema\":\"nat.divergence-probe/1\",\
                \"backend\" : \"candle-cuda\", \"dtype\":\"bf16\",\
                \"perf\":{\"tokens_per_second\":71098.0} }";
    let w = verify(&sign(KEY_A, body)).expect("odd formatting must still verify");
    assert_eq!(w.capability, Capability::Federated);
}

/// The real thing, not a hand-written fixture.
///
/// Captured from nat's `divergence_probe` on the GB10 (CUDA, f32). The two long
/// arrays are truncated for size; every key, nesting level and JSON type is
/// verbatim.
///
/// This is the cross-repo drift guard. nat's document carries five top-level
/// fields this crate does not declare — `job`, `loss`, `params`, `weights`,
/// `zone_shares` — plus `perf.seconds`, and the parser has to ignore them rather
/// than reject a document it does not fully model. If nat renames
/// `self_repeat_identical` or moves `tokens_per_second`, this fails here instead
/// of failing silently against a live fleet.
const REAL_GB10_CUDA_PROBE: &str = r#"{
        "arch": "aarch64",
        "backend": "candle-cuda",
        "dtype": "f32",
        "job": {
            "batch": 16,
            "d": 96,
            "lr": 0.003,
            "merge_floor": 0.01,
            "seed": 20260730,
            "seq": 64,
            "tau": 1.0,
            "vocab": 1024,
            "windows": 384
        },
        "loss": {
            "after": 5.391067981719971,
            "before": 6.933517932891846
        },
        "os": "linux",
        "params": 365735,
        "perf": {
            "seconds": 0.345664311,
            "tokens_per_second": 71097.88085701448
        },
        "schema": "nat.divergence-probe/1",
        "self_repeat_identical": true,
        "weights": {
            "global_l2": 44.64241517959181,
            "per_tensor": [
                {
                    "l2": 24.45722491238502,
                    "n": 98304,
                    "name": "embedding.weight"
                },
                {
                    "l2": 1.7026568429045779,
                    "n": 1024,
                    "name": "readout.bias"
                }
            ],
            "probe_points": [
                0.08215372264385223,
                -0.05330904945731163,
                0.138675257563591
            ],
            "q16_commitment": "075fb3e73b58ce8831df0ca99769064bfb6561b28276d262cc2c91a798145271"
        },
        "zone_shares": {
            "CB": 0.3802832365036011,
            "CX": 0.0027654378209263086,
            "HP": 0.0023936908692121506,
            "PF": 0.0026590824127197266,
            "SM": 0.6118985414505005
        }
    }"#;

#[test]
fn the_parser_accepts_what_nat_actually_emits() {
    let w = verify(&sign(KEY_A, REAL_GB10_CUDA_PROBE)).expect("real probe must verify");
    assert_eq!(w.backend, "candle-cuda");
    assert_eq!(w.dtype, "f32");
    assert!((w.tokens_per_second - 71_097.88).abs() < 0.1);
    // f32 on CUDA, self-repeating and fast: this is exactly ladder hardware now.
    assert_eq!(w.capability, Capability::H01);
}
