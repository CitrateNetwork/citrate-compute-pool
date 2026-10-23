//! Each coordinator-protocol digest carries its own domain tag. A worker signs
//! these with the key it also signs chain transactions with, so a digest of one
//! message type must never equal a digest of another (or a bare keccak of the
//! payload). The coordinator verifies with the same functions, so removing a tag
//! on both sides would pass every round-trip test; these golden recomputations
//! pin the exact pre-images.

use citrate_training_worker::coordinator_protocol::{
    attestation_digest, heartbeat_digest, lease_digest, submission_digest, JobId,
    HEARTBEAT_MESSAGE, LEASE_MESSAGE, REGISTER_MESSAGE,
};
use sha3::{Digest, Keccak256};

fn keccak(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Keccak256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

const PROBE: &str = r#"{"tok_s":42,"gpu":"probe"}"#;
const TS: u64 = 1_760_000_000_123;

#[test]
fn domain_tags_are_distinct_and_versioned() {
    let tags = [
        REGISTER_MESSAGE,
        LEASE_MESSAGE,
        HEARTBEAT_MESSAGE,
        b"citrate-training-submission/1",
    ];
    for (i, a) in tags.iter().enumerate() {
        assert!(
            a.starts_with(b"citrate-training-") && a.ends_with(b"/1"),
            "tag {:?}",
            String::from_utf8_lossy(a)
        );
        for b in &tags[i + 1..] {
            assert_ne!(a, b, "two message types share a domain tag");
        }
    }
}

#[test]
fn attestation_digest_is_domain_separated() {
    let want = keccak(&[
        REGISTER_MESSAGE,
        b"\n",
        &TS.to_be_bytes(),
        b"\n",
        PROBE.as_bytes(),
    ]);
    assert_eq!(attestation_digest(PROBE, TS), want);
    assert_ne!(
        attestation_digest(PROBE, TS),
        keccak(&[PROBE.as_bytes()]),
        "must not be a bare keccak of the probe"
    );
}

#[test]
fn lease_heartbeat_and_submission_digests_are_domain_separated() {
    let job = JobId("h01-64m-nat-seed2".into());
    assert_eq!(
        lease_digest(TS),
        keccak(&[LEASE_MESSAGE, b"\n", &TS.to_be_bytes()])
    );
    assert_eq!(
        heartbeat_digest(&job, TS),
        keccak(&[
            HEARTBEAT_MESSAGE,
            b"\n",
            job.0.as_bytes(),
            b"\n",
            &TS.to_be_bytes()
        ])
    );
    assert_eq!(
        submission_digest(&job, PROBE),
        keccak(&[
            b"citrate-training-submission/1\n",
            job.0.as_bytes(),
            b"\n",
            PROBE.as_bytes()
        ])
    );
    // No two message types collide on the same inputs.
    let ds = [
        attestation_digest(PROBE, TS),
        lease_digest(TS),
        heartbeat_digest(&job, TS),
        submission_digest(&job, PROBE),
    ];
    for (i, a) in ds.iter().enumerate() {
        for b in &ds[i + 1..] {
            assert_ne!(a, b);
        }
    }
}
