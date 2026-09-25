//! The wire vocabulary shared by a worker and the training coordinator.
//!
//! This lives in the **worker** crate for a structural reason: the coordinator
//! already depends on this crate for wallets and recoverable signatures, so
//! defining the protocol here is the only placement that gives both sides one
//! copy without a dependency cycle or a third crate.
//!
//! ## Why the digests must be defined once
//!
//! Each digest below is the exact preimage one side signs and the other
//! recovers. Two hand-rolled implementations of "the obvious concatenation"
//! agree right up until one of them changes a separator — at which point every
//! honest submission fails to authenticate, and the failure presents as a key
//! problem rather than an encoding one. That is a genuinely nasty bug to chase,
//! so neither side is permitted its own copy.
//!
//! Each is domain-separated. A worker signs coordinator messages with the same
//! key it signs chain transactions with, so a message that could be reinterpreted
//! as another message type would be a real vulnerability rather than an
//! untidiness.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

/// Stable identifier for a unit of work. Human-readable on purpose: these appear
/// in logs a member reads while their machine works for two days, and
/// `h01-64m-nat-seed2` explains itself where a UUID does not.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a machine has to be to receive a job. Cumulative: an H-01 machine can do
/// everything below it, modelled as an ordering so a job states one requirement
/// and cannot accidentally be written to exclude better hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    /// Divergence mapping, and anything else where heterogeneity is the point.
    Probe,
    /// Federated co-training. Needs a real accelerator.
    Federated,
    /// The H-01 ablation ladder. Needs bf16 CUDA and a device that reproduces
    /// itself, because the ladder *compares* runs: a numerics difference between
    /// two machines is indistinguishable from the effect being measured.
    H01,
}

impl Capability {
    pub fn satisfies(self, required: Capability) -> bool {
        self >= required
    }
}

/// A unit of work as published by the coordinator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSpec {
    pub id: JobId,
    pub requires: Capability,
    /// What the worker actually runs. Opaque to the coordinator, which schedules
    /// work rather than interpreting training arguments — so a new job shape does
    /// not need a coordinator release.
    pub payload: serde_json::Value,
    /// How long the lease lasts before the job returns to the pool. Per job,
    /// because the ladder spans six-hour rungs and forty-eight-hour ones and one
    /// timeout cannot serve both.
    pub lease_secs: u64,
    pub max_attempts: u32,
}

impl JobSpec {
    pub fn new(id: impl Into<String>, requires: Capability, payload: serde_json::Value) -> Self {
        Self {
            id: JobId(id.into()),
            requires,
            payload,
            lease_secs: 3600,
            max_attempts: 3,
        }
    }

    pub fn with_lease_secs(mut self, s: u64) -> Self {
        self.lease_secs = s;
        self
    }

    pub fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }
}

pub mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{}", hex::encode(v)))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        hex::decode(s.strip_prefix("0x").unwrap_or(&s)).map_err(serde::de::Error::custom)
    }
}

fn keccak(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Keccak256::new();
    for p in parts {
        h.update(p);
    }
    let mut d = [0u8; 32];
    d.copy_from_slice(&h.finalize());
    d
}

/// `keccak256(probe_json)` — what a worker signs to register.
///
/// Deliberately over the probe bytes **verbatim**, with no envelope. JSON does
/// not round-trip byte-for-byte (key order, number formatting), so a digest over
/// a re-serialization would fail for honest workers whose file is a byte off from
/// what the verifier would have produced.
pub fn attestation_digest(probe_json: &str) -> [u8; 32] {
    keccak(&[probe_json.as_bytes()])
}

/// What a worker signs to ask for work. Domain-separated by `LEASE_MESSAGE`, and
/// bound to the moment it was signed so it is not a permanently-replayable bearer
/// credential (CP-B-002): the coordinator rejects a `timestamp` outside
/// [`LEASE_FRESHNESS_NANOS`] of its own clock and remembers recent
/// (signer, timestamp) pairs to refuse an exact replay inside that window.
pub const LEASE_MESSAGE: &[u8] = b"citrate-training-lease/1";

/// Freshness window for a lease request, in nanoseconds. A captured request is
/// usable only inside this window of the coordinator's clock, not forever. Coarse
/// enough (2 minutes) to tolerate ordinary NTP skew between volunteer machines.
pub const LEASE_FRESHNESS_NANOS: u64 = 120 * 1_000_000_000;

/// Bind the request to `timestamp` (unix nanoseconds). Nanosecond granularity
/// means two honest polls, however close, sign distinct preimages, so the
/// coordinator's replay set never rejects a worker's own next request.
pub fn lease_digest(timestamp: u64) -> [u8; 32] {
    keccak(&[LEASE_MESSAGE, b"\n", &timestamp.to_be_bytes()])
}

/// What a worker signs to keep a lease alive (PBA-L3b-001). Domain-separated
/// from every other digest, bound to the job and to the moment it was signed, so
/// a captured heartbeat can neither be replayed outside the freshness window nor
/// be repurposed to renew a different job or to ask for new work.
pub const HEARTBEAT_MESSAGE: &[u8] = b"citrate-training-heartbeat/1";

/// How far past "now" one heartbeat carries a lease, in seconds (PBA-L3b-001).
///
/// A lease used to last its whole `JobSpec::lease_secs` (50 hours for the big
/// ladder rungs), so a key that leased and walked away held the job for two days.
/// Now a lease lives only this long unless its holder keeps heartbeating, capped
/// at the job's `lease_secs` as a hard deadline. A machine that is switched off
/// mid-job releases the work within this window instead of within two days.
pub const LEASE_RENEW_WINDOW_SECS: u64 = 900;

/// How often the worker client heartbeats a running job. Well inside
/// [`LEASE_RENEW_WINDOW_SECS`] so two consecutive missed heartbeats (a flaky
/// domestic link) still do not lose the lease.
pub const HEARTBEAT_INTERVAL_SECS: u64 = 240;

/// `keccak256("citrate-training-heartbeat/1\n" || job_id || "\n" || timestamp_be)`.
pub fn heartbeat_digest(job: &JobId, timestamp: u64) -> [u8; 32] {
    keccak(&[
        HEARTBEAT_MESSAGE,
        b"\n",
        job.0.as_bytes(),
        b"\n",
        &timestamp.to_be_bytes(),
    ])
}

/// `keccak256("citrate-training-submission/1\n" || job_id || "\n" || payload)`.
///
/// The newline separators make the encoding unambiguous for the job ids in use
/// (`h01-64m-nat-seed2`), which contain no newlines.
pub fn submission_digest(job: &JobId, payload: &str) -> [u8; 32] {
    keccak(&[
        b"citrate-training-submission/1\n",
        job.0.as_bytes(),
        b"\n",
        payload.as_bytes(),
    ])
}

/// A signed capability claim, as submitted for registration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attestation {
    /// The probe document exactly as signed — see [`attestation_digest`].
    pub probe_json: String,
    /// 65-byte recoverable secp256k1 signature.
    #[serde(with = "hex_bytes")]
    pub signature: Vec<u8>,
}

/// A request that proves who is asking and when. The signature covers the
/// `timestamp` via [`lease_digest`], so the timestamp cannot be altered without
/// invalidating it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaseRequest {
    /// Unix time in nanoseconds when the request was signed. Defaults to 0 — which
    /// always fails the freshness check — so a body that omits it is refused, not
    /// silently accepted.
    #[serde(default)]
    pub timestamp: u64,
    #[serde(with = "hex_bytes")]
    pub signature: Vec<u8>,
}

/// A signed "still working on it" for a leased job (PBA-L3b-001). Freshness and
/// replay rules are the lease request's: the timestamp must be inside
/// [`LEASE_FRESHNESS_NANOS`] of the coordinator clock and an exact replay is
/// refused.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub job: JobId,
    /// Unix nanoseconds when signed. Defaults to 0, which always fails the
    /// freshness check.
    #[serde(default)]
    pub timestamp: u64,
    #[serde(with = "hex_bytes")]
    pub signature: Vec<u8>,
}

/// The coordinator's answer to a heartbeat: the lease's new expiry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub expires_at: u64,
}

/// A result as returned by a worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignedSubmission {
    pub job: JobId,
    /// Opaque to the coordinator: it records outcomes rather than interpreting
    /// training output. Verifying the *content* belongs to the challenge path,
    /// which can re-run the work; the coordinator can only establish who said it.
    pub payload: String,
    #[serde(with = "hex_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RegisterResponse {
    pub worker: String,
    pub capability: Capability,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_is_cumulative() {
        assert!(Capability::H01.satisfies(Capability::Probe));
        assert!(Capability::H01.satisfies(Capability::Federated));
        assert!(Capability::Federated.satisfies(Capability::Probe));
    }

    #[test]
    fn a_lesser_machine_does_not_satisfy_a_greater_requirement() {
        assert!(!Capability::Probe.satisfies(Capability::Federated));
        assert!(!Capability::Federated.satisfies(Capability::H01));
    }

    /// The wire form is what the coordinator's JSON says. If this changes, every
    /// deployed worker stops being understood.
    #[test]
    fn capability_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Capability::H01).unwrap(), "\"h01\"");
        assert_eq!(
            serde_json::to_string(&Capability::Federated).unwrap(),
            "\"federated\""
        );
        assert_eq!(
            serde_json::to_string(&Capability::Probe).unwrap(),
            "\"probe\""
        );
    }

    /// Each digest must be distinct from the others over the same input, so no
    /// message type can be reinterpreted as another.
    #[test]
    fn the_four_digests_are_domain_separated_from_each_other() {
        let job = JobId("x".into());
        let a = attestation_digest("x");
        let l = lease_digest(0);
        let s = submission_digest(&job, "");
        let h = heartbeat_digest(&job, 0);
        assert_ne!(a, l);
        assert_ne!(a, s);
        assert_ne!(l, s);
        assert_ne!(h, a);
        assert_ne!(h, l);
        assert_ne!(h, s);
    }

    /// A submission digest must not collide with the naive concatenation an
    /// independent implementer would reach for first.
    #[test]
    fn the_submission_digest_is_domain_separated() {
        let job = JobId("j1".into());
        assert_ne!(
            submission_digest(&job, "p"),
            keccak(&[job.0.as_bytes(), b"\n", b"p"])
        );
    }

    /// Job id and payload must not trade characters across the separator and
    /// produce the same digest.
    #[test]
    fn the_job_and_payload_fields_cannot_be_confused_for_each_other() {
        assert_ne!(
            submission_digest(&JobId("a".into()), "b"),
            submission_digest(&JobId("a\nb".into()), "")
        );
    }

    #[test]
    fn hex_signature_fields_round_trip_with_and_without_0x() {
        let with = serde_json::from_str::<LeaseRequest>(r#"{"signature":"0xdeadbeef"}"#).unwrap();
        let without = serde_json::from_str::<LeaseRequest>(r#"{"signature":"deadbeef"}"#).unwrap();
        assert_eq!(with.signature, vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(with.signature, without.signature);
        assert!(serde_json::to_string(&with).unwrap().contains("0xdeadbeef"));
    }
}
