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
    /// Held by a worker until `expires_at` (unix seconds). Heartbeats move
    /// `expires_at` forward, never past `deadline` (PBA-L3b-001). A record
    /// written before `deadline` existed deserialises with `deadline == 0`,
    /// which is read as "no extension beyond `expires_at`".
    Leased {
        worker: H160,
        expires_at: u64,
        #[serde(default)]
        deadline: u64,
    },
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
    /// Where the registration came from (client IP, IPv6 folded to its /64).
    /// Used only to cap how many identities one source may hold (PBA-L3b-001).
    #[serde(default)]
    pub source: String,
    /// Leases this worker let expire without submitting, since its last
    /// accepted submission.
    #[serde(default)]
    pub noshows: u32,
    /// No new lease before this unix second. Set when a lease lapses.
    #[serde(default)]
    pub cooldown_until: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    pub jobs: BTreeMap<JobId, JobRecord>,
    pub workers: BTreeMap<H160, WorkerRecord>,
    /// Operator policy. Configuration, not state: never persisted, supplied at
    /// boot (see [`crate::api::Coordinator::open_with`]).
    #[serde(skip)]
    pub policy: Policy,
}

/// How many unexpired leases one identity may hold at once (PBA-L3b-001). A
/// worker trains one job at a time, so one is all an honest machine needs, and
/// it stops a single key from draining the whole queue with repeated polls.
pub const MAX_LEASES_PER_WORKER: usize = 1;

/// Default cap on distinct identities registered from one source (PBA-L3b-001).
/// Keys are free, so without this one host can mint a fresh key for every lease
/// cycle and never be excluded. Sized for a household or a small lab behind one
/// NAT; an operator with a bigger site raises it
/// (`CITRATE_COORDINATOR_MAX_WORKERS_PER_SOURCE`) or vouches the machines.
pub const DEFAULT_MAX_WORKERS_PER_SOURCE: usize = 16;

/// A source-slot whose worker has not been seen for this long (and holds no
/// lease) may be reclaimed by a new registration from the same source, so a
/// member who re-keys or retires a machine is not locked out forever.
pub const SOURCE_SLOT_STALE_SECS: u64 = 86_400;

/// First no-show cool-down, doubled per consecutive no-show up to
/// [`NOSHOW_BACKOFF_MAX_SECS`] (PBA-L3b-001).
pub const NOSHOW_BACKOFF_BASE_SECS: u64 = 3_600;
pub const NOSHOW_BACKOFF_MAX_SECS: u64 = 7 * 86_400;

/// Operator-supplied admission policy (PBA-L3b-001).
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    /// Addresses the operator has vetted for top-tier (H-01) work.
    ///
    /// The probe that decides a machine's tier is written by the machine
    /// itself, and nothing on the coordinator can re-run it, so a self-reported
    /// H-01 is a claim, not evidence. Anyone could mint a key, claim H-01 and
    /// lease the ladder. The top tier is therefore granted only to vouched
    /// addresses; an unvouched H-01 claim is treated as `Federated`.
    pub trusted_h01: BTreeSet<H160>,
    /// See [`DEFAULT_MAX_WORKERS_PER_SOURCE`].
    pub max_workers_per_source: usize,
    /// See `coordinator_protocol::LEASE_RENEW_WINDOW_SECS`.
    pub lease_window_secs: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            trusted_h01: BTreeSet::new(),
            max_workers_per_source: DEFAULT_MAX_WORKERS_PER_SOURCE,
            lease_window_secs:
                citrate_training_worker::coordinator_protocol::LEASE_RENEW_WINDOW_SECS,
        }
    }
}

impl Policy {
    /// The capability the coordinator actually grants for a claimed one.
    pub fn effective_capability(&self, id: &H160, claimed: Capability) -> Capability {
        if claimed == Capability::H01 && !self.trusted_h01.contains(id) {
            return Capability::Federated;
        }
        claimed
    }

    pub fn is_trusted(&self, id: &H160) -> bool {
        self.trusted_h01.contains(id)
    }
}

/// Parse a comma-separated list of `0x` addresses (whitespace and empty entries
/// ignored). Any malformed entry is an error, so a typo in the operator's list
/// fails the boot instead of silently dropping a machine.
pub fn parse_address_list(raw: &str) -> Result<BTreeSet<H160>, String> {
    raw.split(',')
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(|a| {
            let hex_part = a.strip_prefix("0x").unwrap_or(a);
            let bytes = hex::decode(hex_part).map_err(|e| format!("{a:?}: {e}"))?;
            if bytes.len() != 20 {
                return Err(format!("{a:?}: expected 20 bytes, got {}", bytes.len()));
            }
            Ok(H160::from_slice(&bytes))
        })
        .collect()
}

/// Cool-down after the `n`th consecutive no-show (n >= 1).
pub fn noshow_backoff(n: u32) -> u64 {
    let doublings = n.saturating_sub(1).min(20);
    NOSHOW_BACKOFF_BASE_SECS
        .saturating_mul(1u64 << doublings)
        .min(NOSHOW_BACKOFF_MAX_SECS)
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
    #[error("worker let a lease lapse; no new lease before {until}")]
    CoolingDown { until: u64 },
    #[error("worker already holds the maximum number of leases")]
    AtLeaseCap,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum RegisterError {
    #[error("too many identities registered from this source")]
    SourceFull,
    #[error("the worker registry is full of active workers")]
    RegistryFull,
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

    /// Register or refresh a worker, returning the capability granted.
    ///
    /// Re-registering is normal: a member who upgrades a GPU re-probes, and the
    /// new measurement replaces the old. A refresh never grows the map and is
    /// never refused. A *new* identity is admitted only if its source has room
    /// (PBA-L3b-001) and the registry has room (CP-B-003).
    pub fn register(
        &mut self,
        w: &RegisteredWorker,
        source: &str,
        now: u64,
    ) -> Result<Capability, RegisterError> {
        let granted = self.policy.effective_capability(&w.id, w.capability);
        if let Some(e) = self.workers.get_mut(&w.id) {
            e.capability = w.capability;
            e.backend = w.backend.clone();
            e.dtype = w.dtype.clone();
            e.tokens_per_second = w.tokens_per_second;
            e.last_seen = now;
            return Ok(granted);
        }

        let holders = self.leaseholders(now);
        if !self.policy.is_trusted(&w.id) {
            self.make_room_for_source(source, &holders, now)?;
        }
        // CP-B-003: bound the map. Evict the least-recently-seen worker that is
        // neither vouched nor mid-job; an honest polling worker refreshes
        // `last_seen` on every poll, so it is never the victim of a flood.
        if self.workers.len() >= MAX_WORKERS {
            let evict = self
                .workers
                .iter()
                .filter(|(id, _)| !self.policy.is_trusted(id) && !holders.contains(*id))
                .min_by_key(|(_, r)| r.last_seen)
                .map(|(id, _)| *id)
                .ok_or(RegisterError::RegistryFull)?;
            self.workers.remove(&evict);
        }
        self.workers.insert(
            w.id,
            WorkerRecord {
                capability: w.capability,
                backend: w.backend.clone(),
                dtype: w.dtype.clone(),
                tokens_per_second: w.tokens_per_second,
                registered_at: now,
                last_seen: now,
                source: source.to_string(),
                noshows: 0,
                cooldown_until: 0,
            },
        );
        Ok(granted)
    }

    /// PBA-L3b-001: one source may hold at most `max_workers_per_source`
    /// unvouched identities. When it is at the cap, a stale slot (unseen for
    /// [`SOURCE_SLOT_STALE_SECS`], no live lease) is reclaimed; otherwise the new
    /// identity is refused.
    fn make_room_for_source(
        &mut self,
        source: &str,
        holders: &BTreeSet<H160>,
        now: u64,
    ) -> Result<(), RegisterError> {
        let from_source: Vec<(H160, u64)> = self
            .workers
            .iter()
            .filter(|(id, r)| r.source == source && !self.policy.is_trusted(id))
            .map(|(id, r)| (*id, r.last_seen))
            .collect();
        if from_source.len() < self.policy.max_workers_per_source {
            return Ok(());
        }
        let stale = from_source
            .iter()
            .filter(|(id, seen)| {
                now.saturating_sub(*seen) >= SOURCE_SLOT_STALE_SECS && !holders.contains(id)
            })
            .min_by_key(|(_, seen)| *seen)
            .map(|(id, _)| *id)
            .ok_or(RegisterError::SourceFull)?;
        self.workers.remove(&stale);
        Ok(())
    }

    /// Workers holding a lease that is still live at `now`.
    fn leaseholders(&self, now: u64) -> BTreeSet<H160> {
        self.jobs
            .values()
            .filter_map(|r| match r.status {
                JobStatus::Leased {
                    worker, expires_at, ..
                } if expires_at > now => Some(worker),
                _ => None,
            })
            .collect()
    }

    /// Return expired leases to the pool. Called before every assignment and on a
    /// timer, so a machine that is switched off mid-job does not strand its work.
    pub fn expire_leases(&mut self, now: u64) -> Vec<JobId> {
        let mut freed = Vec::new();
        for (id, rec) in self.jobs.iter_mut() {
            let JobStatus::Leased {
                worker, expires_at, ..
            } = rec.status
            else {
                continue;
            };
            if expires_at > now {
                continue;
            }
            // CP-B-001: an expired lease is a no-show — the worker leased and
            // never submitted. That is not evidence the job itself is bad, and it
            // must not permanently remove the job from circulation. Registration
            // is unauthenticated, so a handful of throwaway keys could otherwise
            // lease-and-expire a job into a terminal `Quarantined` state with no
            // recovery path (each fresh key evades `failed_by`, so `attempts`
            // marched to `max_attempts`). Always return the job to the pool,
            // recording `failed_by` so the same key is not re-offered it. Terminal
            // quarantine is reserved for a genuine failure signal, which the
            // unauthenticated no-show path cannot forge.
            rec.failed_by.insert(worker);
            rec.status = JobStatus::Pending;
            freed.push(id.clone());
            // PBA-L3b-001: a lapsed lease also costs the worker. A key that
            // leases and walks away waits out a doubling cool-down before it
            // may lease anything again.
            if let Some(w) = self.workers.get_mut(&worker) {
                w.noshows = w.noshows.saturating_add(1);
                w.cooldown_until = now.saturating_add(noshow_backoff(w.noshows));
            }
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
        let rec = self
            .workers
            .get_mut(&worker)
            .ok_or(LeaseError::UnknownWorker)?;
        // PBA-L3b-001: a signed poll is proof of life, work or no work. Without
        // this an idle honest worker aged out and a registration flood evicted it.
        rec.last_seen = now;
        let claimed = rec.capability;

        self.expire_leases(now);

        let until = self.workers.get(&worker).map_or(0, |w| w.cooldown_until);
        if until > now {
            return Err(LeaseError::CoolingDown { until });
        }
        let held = self
            .jobs
            .values()
            .filter(|r| matches!(r.status, JobStatus::Leased { worker: h, .. } if h == worker))
            .count();
        if held >= MAX_LEASES_PER_WORKER {
            return Err(LeaseError::AtLeaseCap);
        }
        let cap = self.policy.effective_capability(&worker, claimed);

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

        let window = self.policy.lease_window_secs;
        let rec = self
            .jobs
            .get_mut(&pick)
            .ok_or(LeaseError::NothingAvailable)?;
        rec.attempts += 1;
        // PBA-L3b-001: the lease lives one heartbeat window, extendable up to
        // the job's own `lease_secs` by heartbeats (see [`State::renew`]).
        let deadline = now.saturating_add(rec.spec.lease_secs);
        rec.status = JobStatus::Leased {
            worker,
            expires_at: now.saturating_add(rec.spec.lease_secs.min(window)),
            deadline,
        };
        Ok(rec.spec.clone())
    }

    /// Extend a live lease by one heartbeat window, never past its deadline.
    /// Returns the new expiry. Only the leaseholder may renew, and only while
    /// the lease is still live: a lapsed lease has already gone back to the
    /// pool, and renewing it would take it from whoever picked it up.
    pub fn renew(&mut self, worker: H160, job: &JobId, now: u64) -> Result<u64, SubmitError> {
        let window = self.policy.lease_window_secs;
        let rec = self.jobs.get_mut(job).ok_or(SubmitError::UnknownJob)?;
        let JobStatus::Leased {
            worker: holder,
            expires_at,
            deadline,
        } = rec.status
        else {
            return Err(SubmitError::NotLeased);
        };
        if holder != worker {
            return Err(SubmitError::NotLeaseholder);
        }
        if expires_at <= now {
            return Err(SubmitError::LeaseExpired {
                expired_at: expires_at,
                now,
            });
        }
        let hard = if deadline == 0 { expires_at } else { deadline };
        let extended = now.saturating_add(window).min(hard).max(expires_at);
        rec.status = JobStatus::Leased {
            worker,
            expires_at: extended,
            deadline: hard,
        };
        if let Some(w) = self.workers.get_mut(&worker) {
            w.last_seen = now;
        }
        Ok(extended)
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
                ..
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
                    // A delivered result clears the no-show record.
                    w.noshows = 0;
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
