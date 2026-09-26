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

use citrate_training_worker::coordinator_protocol::attestation_digest;
use citrate_training_worker::wallet::Wallet;
use ethereum_types::H160;
use serde::{Deserialize, Serialize};

use crate::job::Capability;

// Re-exported, never redefined: the worker constructs these and this crate
// verifies them, so a second definition here could drift from the one on the wire.
pub use citrate_training_worker::coordinator_protocol::Attestation;

/// Throughput floor separating accelerator-class from CPU-class on the probe's
/// fixed job, in tokens/second.
///
/// Lowered from 15,000 once real fleet data arrived. The original was
/// interpolated between the GB10's two backends (CPU 7,786, CUDA 71,098) and
/// happened to exclude the first real Apple machine to report — an M2 Max at
/// 13,472 tok/s, which is unambiguously an accelerator. 10,000 sits above every
/// measured CPU (7,786 and 7,137) and below every measured GPU.
///
/// The alf-web compute door carries the identical constant.
pub const ACCELERATOR_TOK_S: f64 = 10_000.0;

/// Upper plausibility bound on a probe's self-declared throughput, in
/// tokens/second on the fixed divergence-probe job.
///
/// `tokens_per_second` is a field the registering worker writes itself. The
/// fastest hardware measured on this job is the GB10 at ~71,098 tok/s; this bound
/// sits well above that and above any accelerator we realistically expect on so
/// small a model. A claim above it is not a measurement, it is a fabrication
/// aimed at minting the top tier, so [`capability_of`] treats it as unverified
/// (`Probe`) — revised upward only when real hardware actually reports there, the
/// same evidence-driven policy `ACCELERATOR_TOK_S` follows.
///
/// This bounds the *implausible* end of self-declaration; it does not make the
/// self-reported capability sound on its own (a value inside the envelope is
/// still unverified). Closing that fully needs a coordinator-issued challenge —
/// see CP-B-001.
pub const MAX_PLAUSIBLE_TOK_S: f64 = 250_000.0;

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
    // CP-B-001: `tokens_per_second` is attacker-written. A value above any
    // measured or realistically-expected result on the fixed probe job is a
    // fabrication, not a measurement — cap such a claim at `Probe` so it cannot be
    // used to mint the top tier. (This bounds only the implausible end; a value
    // inside the envelope is still self-declared, which a challenge must close.)
    if p.perf.tokens_per_second > MAX_PLAUSIBLE_TOK_S {
        return Capability::Probe;
    }
    match (p.backend.as_str(), p.dtype.as_str()) {
        // f32, on any accelerator that reproduces itself.
        //
        // The dtype requirement moved from bf16 to f32 deliberately. bf16 is
        // ~1.8x faster on CUDA, but it is NOT run-to-run deterministic on Metal
        // (measured, 3/3 runs differ), so a bf16 ladder is one CUDA machine's
        // ladder by construction. f32 self-repeats on every backend measured,
        // and it is also the only dtype our cross-backend divergence data covers
        // — the settlement tolerance is validated for f32 and for nothing else.
        ("candle-cuda", "f32") | ("candle-metal", "f32") => Capability::H01,
        // Any other accelerator dtype (notably bf16) can co-train, but must not
        // be given ladder work: unverifiable on Metal, and unmeasured for
        // divergence everywhere.
        ("candle-cuda", _) | ("candle-metal", _) => Capability::Federated,
        _ => Capability::Probe,
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

    let id = Wallet::recover_address(
        &attestation_digest(&a.probe_json, a.timestamp),
        &a.signature,
    )
    .map_err(|_| AttestError::BadSignature)?;

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
