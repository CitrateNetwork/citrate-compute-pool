// Tests for the real federated signer.
//
// The property under test throughout: a contribution's signature must bind the
// claimed identity to the exact bytes signed, and nothing else may verify. This
// is the gate on the reward path — `gather_and_aggregate` drops a contribution
// whose signature does not verify, and a contribution that DOES verify is paid.

use super::*;
use nat_federated::SignedContribution;
use nat_train::StepContribution;
use nat_types::Q16;

// Anvil account #0 — a well-known throwaway. NOT a real key; used only so the
// derived address is deterministic across runs.
const KEY_A: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
// Anvil account #1.
const KEY_B: &str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";

fn signer(hex: &str) -> WalletSigner {
    WalletSigner::new(Wallet::from_hex(hex).expect("load test key"))
}

fn contribution() -> StepContribution {
    StepContribution {
        compute_metered: Q16::from_f32(4.0),
        data_quality: Q16::from_f32(0.5),
        tokens: 1024,
        provenance_hash: "0xprov".into(),
    }
}

// ── Identity ─────────────────────────────────────────────────────────────

/// `node_id` IS the signing address. That equivalence is what removes the
/// roster: there is no separate name that has to be mapped to a key.
#[test]
fn the_node_id_is_the_signing_address() {
    let s = signer(KEY_A);
    let expected = format!("0x{}", hex::encode(Wallet::from_hex(KEY_A).unwrap().address().as_bytes()));
    assert_eq!(s.node_id(), expected);
}

/// Lowercased, so a mixed-case (EIP-55) rendering of the same address is not a
/// different `node_id`. The gather compares this as a string and settles on it.
#[test]
fn the_node_id_is_lowercase_hex() {
    let s = signer(KEY_A);
    assert!(s.node_id().starts_with("0x"));
    assert_eq!(s.node_id(), s.node_id().to_ascii_lowercase());
}

// ── Round trip ───────────────────────────────────────────────────────────

/// The basic contract: what this signer produces, the verifier accepts.
#[test]
fn a_signature_verifies_against_its_own_node_id() {
    let s = signer(KEY_A);
    let msg = b"the canonical signing message";
    let sig = s.sign(msg).expect("local signing cannot fail");
    assert!(RecoveringVerifier.verify(s.node_id(), msg, &sig));
}

/// End to end through the type the gather actually consumes.
#[test]
fn a_signed_contribution_verifies_end_to_end() {
    let s = signer(KEY_A);
    let c = SignedContribution::create(&s, contribution(), "manifest-hash", "trace-hash")
        .expect("local signing cannot fail");

    assert_eq!(c.node_id, s.node_id());
    assert!(RecoveringVerifier.verify(&c.node_id, &c.message(), &c.signature));
}

// ── What must NOT verify ─────────────────────────────────────────────────

/// A tampered field changes the canonical message, so the signature no longer
/// matches. This is the one that matters for money: `compute_metered` feeds
/// `reward_weight` directly, so inflating it must invalidate the signature.
#[test]
fn inflating_the_claimed_compute_breaks_the_signature() {
    let s = signer(KEY_A);
    let mut c = SignedContribution::create(&s, contribution(), "m", "t")
        .expect("sign");
    assert!(RecoveringVerifier.verify(&c.node_id, &c.message(), &c.signature));

    c.contribution.compute_metered = Q16::from_f32(4000.0);
    assert!(
        !RecoveringVerifier.verify(&c.node_id, &c.message(), &c.signature),
        "a node must not be able to inflate its own reward weight after signing"
    );
}

/// Tampering with the corpus binding must also break it — the manifest hash is
/// what ties a reward claim to auditable data.
#[test]
fn swapping_the_manifest_hash_breaks_the_signature() {
    let s = signer(KEY_A);
    let mut c = SignedContribution::create(&s, contribution(), "real-manifest", "t")
        .expect("sign");
    c.manifest_hash = "some-other-manifest".into();
    assert!(!RecoveringVerifier.verify(&c.node_id, &c.message(), &c.signature));
}

/// A valid signature by B, presented as A. It recovers to B's address, which is
/// not the claimed identity — so it fails. This is the impersonation case the
/// roster used to guard, now handled by arithmetic.
#[test]
fn one_node_cannot_sign_as_another() {
    let a = signer(KEY_A);
    let b = signer(KEY_B);
    let msg = b"contribution bytes";

    let sig_b = b.sign(msg).expect("sign");
    assert!(RecoveringVerifier.verify(b.node_id(), msg, &sig_b), "B is B");
    assert!(
        !RecoveringVerifier.verify(a.node_id(), msg, &sig_b),
        "B's signature must not verify as A"
    );
}

/// A signature over DIFFERENT bytes must not verify, even from the right node.
/// Without this a captured signature could be replayed onto another message.
#[test]
fn a_signature_does_not_transfer_to_another_message() {
    let s = signer(KEY_A);
    let sig = s.sign(b"message one").expect("sign");
    assert!(!RecoveringVerifier.verify(s.node_id(), b"message two", &sig));
}

// ── Malformed input is a rejection, not a crash ──────────────────────────

/// The verifier runs over untrusted network input. Every malformed shape must
/// be an ordinary `false` — a panic here would be a remote denial of service on
/// whoever runs the gather.
#[test]
fn malformed_signatures_are_rejected_without_panicking() {
    let s = signer(KEY_A);
    let msg = b"m";
    let good = s.sign(msg).expect("sign");

    for (label, sig) in [
        ("empty", vec![]),
        ("too short", vec![0u8; 64]),
        ("too long", vec![0u8; 66]),
        ("all zeroes", vec![0u8; 65]),
        ("bad recovery id", {
            let mut v = good.clone();
            v[64] = 99;
            v
        }),
        ("flipped byte in r", {
            let mut v = good.clone();
            v[0] ^= 0xFF;
            v
        }),
    ] {
        assert!(
            !RecoveringVerifier.verify(s.node_id(), msg, &sig),
            "{label} must be rejected"
        );
    }
}

/// A `node_id` that is not an address at all is a rejection, not a parse error
/// that escapes.
#[test]
fn a_nonsense_node_id_is_rejected() {
    let s = signer(KEY_A);
    let msg = b"m";
    let sig = s.sign(msg).expect("sign");
    for bad in ["", "not-an-address", "0x", "0xzz"] {
        assert!(!RecoveringVerifier.verify(bad, msg, &sig));
    }
}

/// The verifier carries no state — the property that removes the roster. If it
/// ever gains a key registry, this stops compiling, which is the intent.
#[test]
// `::default()` on a unit struct is the POINT here: it demonstrates that a
// verifier constructed with no configuration, never told about this node,
// still accepts it. Writing `RecoveringVerifier` would prove less.
#[allow(clippy::default_constructed_unit_structs)]
fn the_verifier_is_stateless() {
    let v = RecoveringVerifier;
    let s = signer(KEY_A);
    let msg = b"m";
    let sig = s.sign(msg).expect("sign");
    // A freshly-defaulted verifier, never told about this node, accepts it.
    assert!(RecoveringVerifier::default().verify(s.node_id(), msg, &sig));
    assert!(v.verify(s.node_id(), msg, &sig));
    assert_eq!(std::mem::size_of::<RecoveringVerifier>(), 0, "no state");
}
