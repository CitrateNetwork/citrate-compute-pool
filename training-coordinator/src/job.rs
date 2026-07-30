//! What work exists, and who is allowed to be given it.

use serde::{Deserialize, Serialize};

/// Stable identifier for a unit of work. Human-readable on purpose: these appear
/// in logs a member reads when their machine is doing something for two days, and
/// `h01-64m-nat-seed2` explains itself where a UUID does not.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a machine has to be to receive a job.
///
/// This is the same routing the alf-web compute door shows a member, but **this
/// side is authoritative**: the web form advises, the coordinator decides. The
/// two carry the same thresholds and the same measured constants, and the tests
/// on each side pin them, because a drift between "what the funnel promised" and
/// "what the coordinator grants" would surface as a member being silently
/// unemployable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    /// Divergence mapping and anything else where heterogeneity is the point.
    /// Every registered machine has at least this.
    Probe,
    /// Federated co-training. Needs a real accelerator.
    Federated,
    /// The H-01 ablation ladder. Needs bf16 CUDA and a device that reproduces
    /// itself, because the ladder *compares* runs: a numerics difference between
    /// two machines is indistinguishable from the effect being measured.
    H01,
}

impl Capability {
    /// Capabilities are cumulative — an H-01 machine can do everything below it.
    /// Modelled as an ordering rather than a set so a job states one requirement
    /// and cannot accidentally be written to exclude better hardware.
    pub fn satisfies(self, required: Capability) -> bool {
        self >= required
    }
}

/// A unit of work as published by the coordinator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSpec {
    pub id: JobId,
    /// Minimum capability. A machine below it never sees the job.
    pub requires: Capability,
    /// What the worker actually runs. Opaque here on purpose: the coordinator
    /// schedules work, it does not interpret training arguments, so a new job
    /// shape does not need a coordinator release.
    pub payload: serde_json::Value,
    /// How long a lease lasts before the job returns to the pool. Sized per job
    /// because the ladder spans six-hour rungs and forty-eight-hour ones, and one
    /// timeout cannot serve both.
    pub lease_secs: u64,
    /// Give up after this many failed attempts rather than handing a poisoned job
    /// around the fleet forever.
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
        assert!(!Capability::Probe.satisfies(Capability::H01));
        assert!(!Capability::Federated.satisfies(Capability::H01));
    }

    #[test]
    fn every_capability_satisfies_itself() {
        for c in [Capability::Probe, Capability::Federated, Capability::H01] {
            assert!(c.satisfies(c));
        }
    }
}
