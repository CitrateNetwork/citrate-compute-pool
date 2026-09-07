//! The coordinator's state machine.
//!
//! Pure and synchronous: no clock, no disk, no network. Time enters as a `now`
//! argument, which is what lets lease expiry — the property that matters most on
//! volunteer hardware and is the hardest to observe in production — be tested
//! directly instead of with sleeps.

use std::collections::{BTreeMap, BTreeSet};

use ethereum_types::H160;
use serde::{Deserialize, Serialize};

use crate::attestation::RegisteredWorker;
use crate::job::{Capability, JobId, JobSpec};

/// Where a job is in its life.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum JobStatus {
    /// Available to be leased.
    Pending,
    /// Held by a worker until `expires_at` (unix seconds).
    Leased { worker: H160, expires_at: u64 },
    /// Submitted and accepted.
    Done { worker: H160, at: u64 },
    /// Failed `max_attempts` times. Parked for a human rather than handed around
    /// the fleet forever — a job that three machines could not finish is a bug
    /// report, not a scheduling problem.
    Quarantined { reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    pub spec: JobSpec,
    pub status: JobStatus,
    pub attempts: u32,
    /// Workers whose lease on this job expired. They are not offered it again:
    /// a job that OOMs on an 8 GB card will OOM on the same card tomorrow, and
    /// re-leasing it there burns the fleet's time proving that twice.
    pub failed_by: BTreeSet<H160>,
    /// Set once accepted. The signed result, kept verbatim for audit.
    pub result: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerRecord {
    pub capability: Capability,
    pub backend: String,
    pub dtype: String,
    pub tokens_per_second: f64,
    pub registered_at: u64,
    pub last_seen: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    pub jobs: BTreeMap<JobId, JobRecord>,
    pub workers: BTreeMap<H160, WorkerRecord>,
}

/// Upper bound on the worker map. `/v1/register` is unauthenticated and free, so
/// without a cap every distinct key becomes a permanent `WorkerRecord` and each
/// subsequent request re-serialises and fsyncs the whole growing state file — an
/// unauthenticated party can degrade then kill the coordinator (CP-B-003). A
/// volunteer fleet is tens of machines; this is generously above that and bounds
/// `state.json` at roughly a megabyte. When full, the least-recently-seen worker
/// is evicted (see [`State::register`]).
pub const MAX_WORKERS: usize = 4096;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum LeaseError {
    #[error("worker is not registered")]
    UnknownWorker,
    #[error("no job available for this worker")]
    NothingAvailable,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SubmitError {
    #[error("no such job")]
    UnknownJob,
    #[error("job is not leased to this worker")]
    NotLeaseholder,
    #[error("the lease expired at {expired_at}, it is now {now}")]
    LeaseExpired { expired_at: u64, now: u64 },
    #[error("job is not in a submittable state")]
    NotLeased,
}

impl State {
    pub fn add_job(&mut self, spec: JobSpec) {
        self.jobs.insert(
            spec.id.clone(),
            JobRecord {
                spec,
                status: JobStatus::Pending,
                attempts: 0,
                failed_by: BTreeSet::new(),
                result: None,
            },
        );
    }

    /// Register or refresh a worker. Re-registering is normal — a member who
    /// upgrades a GPU re-probes, and the new measurement replaces the old.
    pub fn register(&mut self, w: &RegisteredWorker, now: u64) {
        let registered_at = self
            .workers
            .get(&w.id)
            .map(|e| e.registered_at)
            .unwrap_or(now);
        // CP-B-003: only a brand-new id grows the map (a refresh is free). When
        // full, evict the least-recently-seen worker so an unauthenticated flood
        // of fresh keys cannot grow `state.json` without bound. An active honest
        // worker refreshes `last_seen` on every lease, so it is never the victim.
        if !self.workers.contains_key(&w.id) && self.workers.len() >= MAX_WORKERS {
            if let Some(evict) = self
                .workers
                .iter()
                .min_by_key(|(_, r)| r.last_seen)
                .map(|(id, _)| *id)
            {
                self.workers.remove(&evict);
            }
        }
        self.workers.insert(
            w.id,
            WorkerRecord {
                capability: w.capability,
                backend: w.backend.clone(),
                dtype: w.dtype.clone(),
                tokens_per_second: w.tokens_per_second,
                registered_at,
                last_seen: now,
            },
        );
    }

    /// Return expired leases to the pool. Called before every assignment and on a
    /// timer, so a machine that is switched off mid-job does not strand its work.
    pub fn expire_leases(&mut self, now: u64) -> Vec<JobId> {
        let mut freed = Vec::new();
        for (id, rec) in self.jobs.iter_mut() {
            let JobStatus::Leased { worker, expires_at } = rec.status else {
                continue;
            };
            if expires_at > now {
                continue;
            }
            rec.failed_by.insert(worker);
            rec.status = if rec.attempts >= rec.spec.max_attempts {
                JobStatus::Quarantined {
                    reason: format!("{} attempts expired without a submission", rec.attempts),
                }
            } else {
                JobStatus::Pending
            };
            freed.push(id.clone());
        }
        freed
    }

    /// Lease the most demanding job this worker is able to do.
    ///
    /// Most-demanding-first is deliberate. Capable machines are the scarce
    /// resource: if an H-01 machine takes a probe job while an H-01 job waits,
    /// the ladder stalls behind work that any laptop could have done. Ties break
    /// on job id so the assignment is reproducible.
    pub fn lease(&mut self, worker: H160, now: u64) -> Result<JobSpec, LeaseError> {
        let cap = self
            .workers
            .get(&worker)
            .map(|w| w.capability)
            .ok_or(LeaseError::UnknownWorker)?;

        self.expire_leases(now);

        let pick = self
            .jobs
            .values()
            .filter(|r| r.status == JobStatus::Pending)
            .filter(|r| cap.satisfies(r.spec.requires))
            .filter(|r| !r.failed_by.contains(&worker))
            .max_by(|a, b| {
                a.spec
                    .requires
                    .cmp(&b.spec.requires)
                    // BTreeMap iterates ascending by id; `max_by` keeps the LAST
                    // maximum, so reverse the tie-break to settle on the first id.
                    .then_with(|| b.spec.id.cmp(&a.spec.id))
            })
            .map(|r| r.spec.id.clone())
            .ok_or(LeaseError::NothingAvailable)?;

        let rec = self.jobs.get_mut(&pick).expect("just selected");
        rec.attempts += 1;
        rec.status = JobStatus::Leased {
            worker,
            expires_at: now + rec.spec.lease_secs,
        };
        if let Some(w) = self.workers.get_mut(&worker) {
            w.last_seen = now;
        }
        Ok(rec.spec.clone())
    }

    /// Accept a result from the worker holding the lease.
    ///
    /// The caller must have already verified the signature and recovered
    /// `worker`; this checks that the recovered signer is the leaseholder. That
    /// split is on purpose — a perfectly valid signature from a worker who does
    /// not hold the lease must still be refused, and conflating "signed" with
    /// "authorised" is how that gets missed.
    pub fn submit(
        &mut self,
        worker: H160,
        job: &JobId,
        result: String,
        now: u64,
    ) -> Result<(), SubmitError> {
        let rec = self.jobs.get_mut(job).ok_or(SubmitError::UnknownJob)?;
        match rec.status {
            JobStatus::Leased {
                worker: holder,
                expires_at,
            } => {
                if holder != worker {
                    return Err(SubmitError::NotLeaseholder);
                }
                if expires_at <= now {
                    return Err(SubmitError::LeaseExpired {
                        expired_at: expires_at,
                        now,
                    });
                }
                rec.status = JobStatus::Done { worker, at: now };
                rec.result = Some(result);
                if let Some(w) = self.workers.get_mut(&worker) {
                    w.last_seen = now;
                }
                Ok(())
            }
            _ => Err(SubmitError::NotLeased),
        }
    }

    pub fn counts(&self) -> Counts {
        let mut c = Counts::default();
        for r in self.jobs.values() {
            match r.status {
                JobStatus::Pending => c.pending += 1,
                JobStatus::Leased { .. } => c.leased += 1,
                JobStatus::Done { .. } => c.done += 1,
                JobStatus::Quarantined { .. } => c.quarantined += 1,
            }
        }
        c.workers = self.workers.len();
        c
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub pending: usize,
    pub leased: usize,
    pub done: usize,
    pub quarantined: usize,
    pub workers: usize,
}

#[cfg(test)]
mod tests {
    include!("state_tests.rs");
}
