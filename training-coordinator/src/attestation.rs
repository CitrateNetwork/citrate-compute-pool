//! How a machine proves what it is.
//!
//! A worker does not *claim* a capability, it *evidences* one. It runs nat's
//! `divergence_probe` — a fixed job on synthetic tokens — and signs the resulting
//! JSON with the same key it transacts with. The coordinator recovers the signing
//! address from the signature, so:
//!
//!   * the worker's identity **is** the recovered address. There is no roster to
//!     distribute and nothing to go stale (the property `RecoveringVerifier` was
//!     built for in `citrate-training-worker`);
//!   * the capability is **derived from a measurement**, not from a field the
//!     worker filled in;
//!   * tampering with the measurement to claim a better tier invalidates the
//!     signature, because the signature covers the probe bytes.
//!
//! What this does NOT prove is that the probe was run on the machine that will
//! run the job. A member could probe a fast box and train on a slow one. That is
//! a real gap and it is deliberately not closed here: the defence against it is
//! the work itself failing verification downstream, not an attestation scheme
//! that cannot be made sound without hardware roots of trust nobody has.

use citrate_training_worker::wallet::Wallet;
use ethereum_types::H160;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

use crate::job::Capability;

/// Throughput floor separating accelerator-class from CPU-class on the probe's
/// fixed job, in tokens/second.
///
/// Sits between the two backends measured on the reference machine (GB10: CPU
/// 7,786 tok/s, CUDA 71,098 tok/s). **Provisional, from a single machine** —
/// revisit once the fleet reports a real distribution instead of an interpolated
/// one. The alf-web compute door carries the identical constant.
pub const ACCELERATOR_TOK_S: f64 = 15_000.0;

pub const PROBE_SCHEMA: &str = "nat.divergence-probe/1";

/// The fields of a `nat.divergence-probe/1` document this crate reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub schema: String,
    pub backend: String,
    pub dtype: String,
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub arch: String,
    pub perf: ProbePerf,
    /// Whether the device reproduced its own result. See [`capability_of`].
    pub self_repeat_identical: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbePerf {
    pub tokens_per_second: f64,
}

/// The capability a probe evidences.
///
/// The self-repeat gate is load-bearing and is checked first. A device that
/// cannot reproduce its own result cannot have its work verified by
/// recomputation, so it must never be given work whose settlement depends on
/// that — however fast it is. Such a machine is not rejected: mapping divergence
/// is exactly what it is good for, and a device that fails self-repeat is a
/// finding worth having.
pub fn capability_of(p: &ProbeReport) -> Capability {
    if !p.self_repeat_identical {
        return Capability::Probe;
    }
    let fast = p.perf.tokens_per_second >= ACCELERATOR_TOK_S;
    if !fast {
        return Capability::Probe;
    }
    match (p.backend.as_str(), p.dtype.as_str()) {
        // H-01 is measured in bf16; an f32 arm is a different experiment.
        ("candle-cuda", "bf16") => Capability::H01,
        ("candle-cuda", _) | ("candle-metal", _) => Capability::Federated,
        _ => Capability::Probe,
    }
}

/// A signed capability claim, as submitted for registration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attestation {
    /// The probe document, verbatim, exactly as signed. Kept as the original
    /// bytes rather than a re-serialization: re-encoding JSON does not round-trip
    /// byte-for-byte (key order, number formatting), and a signature over
    /// re-encoded bytes would fail for honest workers.
    pub probe_json: String,
    /// 65-byte recoverable secp256k1 signature over `keccak256(probe_json)`.
    #[serde(with = "hex_bytes")]
    pub signature: Vec<u8>,
}

pub(crate) mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{}", hex::encode(v)))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        hex::decode(s.strip_prefix("0x").unwrap_or(&s)).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AttestError {
    #[error("probe is not valid JSON: {0}")]
    NotJson(String),
    #[error("expected schema {PROBE_SCHEMA}, got {0:?}")]
    WrongSchema(String),
    #[error("signature does not recover to a valid address")]
    BadSignature,
}

/// A machine that has proved what it is.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisteredWorker {
    /// The recovered signing address. This IS the worker's identity.
    pub id: H160,
    pub capability: Capability,
    pub backend: String,
    pub dtype: String,
    pub tokens_per_second: f64,
}

/// Verify an attestation and derive the worker it registers.
///
/// Every failure is an ordinary `Err`. This runs over unauthenticated network
/// input, so a malformed attestation must be a rejection rather than a panic —
/// a panic here would be a denial of service on the coordinator by anyone who
/// can reach it.
pub fn verify(a: &Attestation) -> Result<RegisteredWorker, AttestError> {
    let v: serde_json::Value =
        serde_json::from_str(&a.probe_json).map_err(|e| AttestError::NotJson(e.to_string()))?;
    let schema = v.get("schema").and_then(|s| s.as_str()).unwrap_or_default();
    if schema != PROBE_SCHEMA {
        return Err(AttestError::WrongSchema(schema.to_string()));
    }
    let probe: ProbeReport =
        serde_json::from_value(v).map_err(|e| AttestError::NotJson(e.to_string()))?;

    let mut h = Keccak256::new();
    h.update(a.probe_json.as_bytes());
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&h.finalize());

    let id =
        Wallet::recover_address(&digest, &a.signature).map_err(|_| AttestError::BadSignature)?;

    Ok(RegisteredWorker {
        id,
        capability: capability_of(&probe),
        backend: probe.backend,
        dtype: probe.dtype,
        tokens_per_second: probe.perf.tokens_per_second,
    })
}

#[cfg(test)]
mod tests {
    include!("attestation_tests.rs");
}
