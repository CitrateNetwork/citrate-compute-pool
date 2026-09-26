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
    /// PBA-L3b-001: source groups (see [`source_group`]) whose worker let a
    /// lease on this job lapse. Like `failed_by`, but it survives key rotation:
    /// a fresh key from the same host or /48 is not offered the job again.
    #[serde(default)]
    pub failed_by_sources: BTreeSet<String>,
    /// After a lapse the job is offered only to vouched workers until this unix
    /// second, so the next unvouched key cannot win the race for it.
    #[serde(default)]
    pub held_until: u64,
    /// Lease expiry as last written to disk (not itself persisted; 0 after a
    /// load, so the first heartbeat is written). See
    /// [`HEARTBEAT_PERSIST_STEP_SECS`].
    #[serde(skip)]
    persisted_expiry: u64,
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
    /// Results this worker has had accepted. Informational only: submissions
    /// are not verified, so this grants no scheduling priority.
    #[serde(default)]
    pub delivered: u32,
    /// Last registration (unix seconds); refreshes are throttled per key.
    #[serde(default)]
    pub last_registration: u64,
    /// Last time this worker asked for work or heartbeated a lease (unix
    /// seconds). A registration alone does not count.
    #[serde(default)]
    pub last_engaged: Option<u64>,
}

/// No-show history of a source group (PBA-L3b-001). Keys are free, a network
/// is not: this is what makes the back-off survive key rotation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceRecord {
    pub noshows: u32,
    pub cooldown_until: u64,
    pub last_seen: u64,
}

/// A key's no-show history, kept after its registry record is evicted or its
/// source slot reclaimed, and restored if it registers again (PBA-L3b-001).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Penalty {
    pub noshows: u32,
    pub cooldown_until: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    pub jobs: BTreeMap<JobId, JobRecord>,
    pub workers: BTreeMap<H160, WorkerRecord>,
    /// PBA-L3b-001: per-source-group no-show history (bounded, see
    /// [`MAX_SOURCES`]).
    #[serde(default)]
    pub sources: BTreeMap<String, SourceRecord>,
    /// PBA-L3b-001: no-show history of keys no longer in `workers` (bounded by
    /// [`MAX_WORKERS`]).
    #[serde(default)]
    pub penalties: BTreeMap<H160, Penalty>,
    /// Operator policy. Configuration, not state: never persisted, supplied at
    /// boot (see [`crate::api::Coordinator::open_with`]).
    #[serde(skip)]
    pub policy: Policy,
    /// Set by every change that must be persisted; bookkeeping such as a
    /// poll refreshing `last_seen` does not set it. See [`State::take_dirty`].
    #[serde(skip)]
    dirty: bool,
    /// (signer, timestamp) of registrations accepted within the freshness
    /// window. Persisted, so a restart does not make them acceptable again.
    #[serde(default)]
    pub recent_registrations: BTreeSet<(H160, u64)>,
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

/// Default cap on live leases held by unvouched workers of one source group
/// (PBA-L3b-001), so one host cannot take the whole queue with many keys.
/// Overridable with `CITRATE_COORDINATOR_MAX_LEASES_PER_SOURCE`.
pub const DEFAULT_MAX_LEASES_PER_SOURCE: usize = 4;

/// How long a lapsed job is reserved for vouched workers. Longer than the
/// worker client's maximum idle poll interval (300 s), so a vouched worker
/// that is polling will see it.
pub const LAPSE_HOLD_SECS: u64 = 900;

/// A vouched worker seen within this many seconds counts as active. A busy
/// vouched worker heartbeats every few minutes, so this only lapses when no
/// vouched machine is running at all.
pub const VOUCHED_ACTIVE_SECS: u64 = 3_600;

/// Longest a lapsed job may stay reserved for vetted workers, whatever its
/// lease length (see [`reservation_cap`]).
pub const RESERVATION_MAX_SECS: u64 = 7 * 86_400;

/// A heartbeat is persisted only once it has moved the lease expiry this far
/// past the last persisted value (or reached the deadline).
pub const HEARTBEAT_PERSIST_STEP_SECS: u64 = 300;

/// Default for [`Policy::register_refresh_secs`].
pub const DEFAULT_REGISTER_REFRESH_SECS: u64 = 60;

/// How long a lapsed job above the probe tier may stay reserved for vetted
/// workers, counted from the lapse.
/// `min(7 days, max(LAPSE_HOLD_SECS, 2 x lease_secs))`.
pub fn reservation_cap(lease_secs: u64) -> u64 {
    lease_secs
        .saturating_mul(2)
        .clamp(LAPSE_HOLD_SECS, RESERVATION_MAX_SECS)
}

/// Bound on the persisted `sources` map.
pub const MAX_SOURCES: usize = MAX_WORKERS;

/// The unit a no-show is charged to (PBA-L3b-001). IPv6 sources arrive folded
/// to a /64 (see `api::source_of`) and are widened to their /48 here, because
/// one site is routinely delegated a whole /48; IPv4 stays per address.
pub fn source_group(source: &str) -> String {
    if let Some(prefix) = source.strip_suffix("::/64") {
        let segs: Vec<&str> = prefix.split(':').collect();
        if segs.len() == 4 {
            return format!("{}:{}:{}::/48", segs[0], segs[1], segs[2]);
        }
    }
    source.to_string()
}

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
    /// addresses.
    pub trusted_h01: BTreeSet<H160>,
    /// Highest tier an unvouched worker may be granted (`Probe` by default;
    /// an operator may open `Federated`, never `H01`). Work above it goes only
    /// to vouched addresses.
    pub open_tier: Capability,
    /// See [`DEFAULT_MAX_WORKERS_PER_SOURCE`].
    pub max_workers_per_source: usize,
    /// See `coordinator_protocol::LEASE_RENEW_WINDOW_SECS`.
    pub lease_window_secs: u64,
    /// See [`DEFAULT_MAX_LEASES_PER_SOURCE`].
    pub max_leases_per_source: usize,
    /// Minimum seconds between two registrations of the same key.
    pub register_refresh_secs: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            trusted_h01: BTreeSet::new(),
            open_tier: Capability::Probe,
            max_workers_per_source: DEFAULT_MAX_WORKERS_PER_SOURCE,
            max_leases_per_source: DEFAULT_MAX_LEASES_PER_SOURCE,
            register_refresh_secs: DEFAULT_REGISTER_REFRESH_SECS,
            lease_window_secs:
                citrate_training_worker::coordinator_protocol::LEASE_RENEW_WINDOW_SECS,
        }
    }
}

impl Policy {
    /// The capability the coordinator actually grants for a claimed one.
    pub fn effective_capability(&self, id: &H160, claimed: Capability) -> Capability {
        if self.trusted_h01.contains(id) {
            return claimed;
        }
        claimed.min(self.open_tier).min(Capability::Federated)
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
    #[error("the current policy no longer allows this worker this job's tier")]
    TierRevoked,
}

impl State {
    pub fn add_job(&mut self, spec: JobSpec) {
        self.dirty = true;
        self.jobs.insert(
            spec.id.clone(),
            JobRecord {
                spec,
                status: JobStatus::Pending,
                attempts: 0,
                failed_by: BTreeSet::new(),
                result: None,
                failed_by_sources: BTreeSet::new(),
                held_until: 0,
                persisted_expiry: 0,
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
            // Only the claimed capability affects scheduling, so only it is
            // worth a save; the descriptive fields ride along with the next one.
            let changed = e.capability != w.capability;
            e.last_registration = now;
            e.capability = w.capability;
            e.backend = w.backend.clone();
            e.dtype = w.dtype.clone();
            e.tokens_per_second = w.tokens_per_second;
            e.last_seen = now;
            self.dirty |= changed;
            return Ok(granted);
        }
        self.dirty = true;

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
            self.forget_worker(&evict, now);
        }
        let penalty = self.penalties.remove(&w.id).unwrap_or_default();
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
                noshows: penalty.noshows,
                cooldown_until: penalty.cooldown_until,
                delivered: 0,
                last_registration: now,
                last_engaged: None,
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
        self.forget_worker(&stale, now);
        Ok(())
    }

    /// Would a registration of a never-seen `id` from `source` be admitted?
    /// Non-mutating, so the API can check it before spending the global
    /// new-identity budget (PBA-L3b-001 verifier finding: a full source must
    /// not be able to drain that budget).
    pub fn admits_new(&self, id: &H160, source: &str, now: u64) -> Result<(), RegisterError> {
        if self.policy.is_trusted(id) {
            return Ok(());
        }
        let holders = self.leaseholders(now);
        let from_source: Vec<(&H160, &WorkerRecord)> = self
            .workers
            .iter()
            .filter(|(id, r)| r.source == source && !self.policy.is_trusted(id))
            .collect();
        if from_source.len() < self.policy.max_workers_per_source {
            return Ok(());
        }
        let reclaimable = from_source.iter().any(|(id, r)| {
            now.saturating_sub(r.last_seen) >= SOURCE_SLOT_STALE_SECS && !holders.contains(id)
        });
        if reclaimable {
            Ok(())
        } else {
            Err(RegisterError::SourceFull)
        }
    }

    /// Drop a worker record, keeping its no-show history (PBA-L3b-001: a key
    /// must not be able to launder its back-off by being evicted and
    /// re-registering).
    fn forget_worker(&mut self, id: &H160, now: u64) {
        let Some(r) = self.workers.remove(id) else {
            return;
        };
        if r.noshows == 0 && r.cooldown_until <= now {
            return;
        }
        if self.penalties.len() >= MAX_WORKERS && !self.penalties.contains_key(id) {
            if let Some(oldest) = self
                .penalties
                .iter()
                .min_by_key(|(_, p)| p.cooldown_until)
                .map(|(k, _)| *k)
            {
                self.penalties.remove(&oldest);
            }
        }
        self.penalties.insert(
            *id,
            Penalty {
                noshows: r.noshows,
                cooldown_until: r.cooldown_until,
            },
        );
    }

    /// Charge a no-show to a source group, bounded like the worker map.
    fn charge_source(&mut self, group: &str, now: u64) {
        if !self.sources.contains_key(group) && self.sources.len() >= MAX_SOURCES {
            let victim = self
                .sources
                .iter()
                .min_by_key(|(_, r)| (r.cooldown_until > now, r.last_seen))
                .map(|(k, _)| k.clone());
            if let Some(v) = victim {
                self.sources.remove(&v);
            }
        }
        let rec = self.sources.entry(group.to_string()).or_default();
        rec.noshows = rec.noshows.saturating_add(1);
        rec.cooldown_until = now.saturating_add(noshow_backoff(rec.noshows));
        rec.last_seen = now;
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
        let mut lapsed_groups = Vec::new();
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
            // PBA-L3b-001: reserve the lapsed job for established workers for a
            // while, so a squatter's next fresh key cannot win it back.
            rec.held_until = now.saturating_add(LAPSE_HOLD_SECS);
            freed.push(id.clone());
            self.dirty = true;
            // PBA-L3b-001: a lapsed lease also costs the worker. A key that
            // leases and walks away waits out a doubling cool-down before it
            // may lease anything again...
            let mut group = None;
            if let Some(w) = self.workers.get_mut(&worker) {
                w.noshows = w.noshows.saturating_add(1);
                w.cooldown_until = now.saturating_add(noshow_backoff(w.noshows));
                if !self.policy.trusted_h01.contains(&worker) {
                    group = Some(source_group(&w.source));
                }
            }
            // ...and so does its source group, which is what survives key
            // rotation: that network is not offered this job again and waits
            // out its own doubling cool-down.
            if let Some(g) = group {
                rec.failed_by_sources.insert(g.clone());
                lapsed_groups.push(g);
            }
        }
        for g in lapsed_groups {
            self.charge_source(&g, now);
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
        rec.last_engaged = Some(now);
        let claimed = rec.capability;
        let group = source_group(&rec.source);

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
        let trusted = self.policy.is_trusted(&worker);
        if !trusted {
            // PBA-L3b-001: the source group's own cool-down and lease cap,
            // which a fresh key from the same network does not escape.
            let until = self.sources.get(&group).map_or(0, |r| r.cooldown_until);
            if until > now {
                return Err(LeaseError::CoolingDown { until });
            }
            let from_group = self
                .jobs
                .values()
                .filter_map(|r| match r.status {
                    JobStatus::Leased { worker: h, .. } => Some(h),
                    _ => None,
                })
                .filter(|h| !self.policy.is_trusted(h))
                .filter(|h| {
                    self.workers
                        .get(h)
                        .is_some_and(|w| source_group(&w.source) == group)
                })
                .count();
            if from_group >= self.policy.max_leases_per_source {
                return Err(LeaseError::AtLeaseCap);
            }
        }
        // Only vouched workers get first claim on a lapsed job: an accepted
        // submission is not verified, so it earns no priority.
        let established = trusted;
        let cap = self.policy.effective_capability(&worker, claimed);

        let pick = self
            .jobs
            .values()
            .filter(|r| r.status == JobStatus::Pending)
            .filter(|r| cap.satisfies(r.spec.requires))
            .filter(|r| !r.failed_by.contains(&worker))
            .filter(|r| trusted || !r.failed_by_sources.contains(&group))
            .filter(|r| established || !self.reserved_for_vouched(r, now))
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
        self.dirty = true;
        rec.status = JobStatus::Leased {
            worker,
            expires_at: now.saturating_add(rec.spec.lease_secs.min(window)),
            deadline,
        };
        Ok(rec.spec.clone())
    }

    /// May `worker` hold a job requiring `requires` under the current policy?
    ///
    /// This is the *operator policy* check only: the worker's own claimed
    /// capability is not consulted. A worker changing its own claim mid-lease
    /// does not release the lease (it runs to its deadline and a lapse is
    /// charged as usual); only a policy change requeues work for free.
    fn tier_allowed(&self, worker: &H160, requires: Capability) -> bool {
        self.policy
            .effective_capability(worker, Capability::H01)
            .satisfies(requires)
    }

    /// Return a job whose lease the current policy no longer allows to the
    /// pool. Not a no-show: the worker is not penalised and may be offered
    /// the job again if the policy allows it later.
    fn revoke_lease(&mut self, job: &JobId) {
        if let Some(rec) = self.jobs.get_mut(job) {
            rec.status = JobStatus::Pending;
            self.dirty = true;
        }
    }

    /// At start: give every lease still inside its deadline a fresh renewal
    /// window.
    ///
    /// Heartbeats are persisted in steps (and cannot arrive while the
    /// coordinator is down), so the expiry on disk may trail the real one;
    /// without this a restart would charge honest workers a no-show.
    pub fn grace_live_leases(&mut self, now: u64) {
        let window = self.policy.lease_window_secs;
        for rec in self.jobs.values_mut() {
            if let JobStatus::Leased {
                worker,
                expires_at,
                deadline,
            } = rec.status
            {
                let hard = if deadline == 0 { expires_at } else { deadline };
                if hard > now {
                    let renewed = now.saturating_add(window).min(hard).max(expires_at);
                    if renewed != expires_at {
                        rec.status = JobStatus::Leased {
                            worker,
                            expires_at: renewed,
                            deadline: hard,
                        };
                        self.dirty = true;
                    }
                }
            }
        }
    }

    /// Whether anything that must be persisted changed since the last call.
    /// Record an accepted registration `(id, timestamp)`; `false` if it was
    /// already accepted. Entries outside the freshness window are pruned.
    pub fn note_registration(&mut self, id: H160, timestamp: u64, now_nanos: u64) -> bool {
        let window = citrate_training_worker::coordinator_protocol::LEASE_FRESHNESS_NANOS;
        self.recent_registrations
            .retain(|(_, ts)| ts.abs_diff(now_nanos) <= window);
        let inserted = self.recent_registrations.insert((id, timestamp));
        self.dirty |= inserted;
        inserted
    }

    /// Was `(id, timestamp)` already accepted? Non-mutating.
    pub fn registration_seen(&self, id: H160, timestamp: u64) -> bool {
        self.recent_registrations.contains(&(id, timestamp))
    }

    /// Mark the state as needing a save (e.g. after a failed save, so the
    /// next request retries it).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    ///
    /// A `true` means the caller is about to save: the lease expiries being
    /// written are recorded, so later heartbeats are measured against them.
    pub fn take_dirty(&mut self) -> bool {
        let dirty = std::mem::take(&mut self.dirty);
        if dirty {
            for rec in self.jobs.values_mut() {
                if let JobStatus::Leased { expires_at, .. } = rec.status {
                    rec.persisted_expiry = expires_at;
                }
            }
        }
        dirty
    }

    /// Revoke every lease the current policy no longer allows (called at
    /// boot, so a policy tightened across a restart applies to work already
    /// out). Returns the requeued jobs.
    pub fn revoke_out_of_policy(&mut self) -> Vec<JobId> {
        let revoked: Vec<JobId> = self
            .jobs
            .iter()
            .filter_map(|(id, r)| match r.status {
                JobStatus::Leased { worker, .. }
                    if !self.tier_allowed(&worker, r.spec.requires) =>
                {
                    Some(id.clone())
                }
                _ => None,
            })
            .collect();
        for id in &revoked {
            self.revoke_lease(id);
        }
        revoked
    }

    /// Is a pending job reserved for vetted workers at `now`? After a lapse
    /// it is for [`LAPSE_HOLD_SECS`]. A job above the probe tier stays
    /// reserved after that while a vetted worker that could take it is engaged
    /// (asked for work or heartbeated within [`VOUCHED_ACTIVE_SECS`]; a
    /// registration alone does not count), its claim satisfies the job and it
    /// is not in `failed_by`, up to [`reservation_cap`] after the lapse, so a
    /// busy vetted fleet is waited for rather than the job going back to the
    /// open pool. Probe jobs only get the plain hold.
    fn reserved_for_vouched(&self, rec: &JobRecord, now: u64) -> bool {
        if rec.held_until == 0 {
            return false;
        }
        if rec.held_until > now {
            return true;
        }
        if rec.spec.requires == Capability::Probe {
            return false;
        }
        let lapsed_at = rec.held_until.saturating_sub(LAPSE_HOLD_SECS);
        if now >= lapsed_at.saturating_add(reservation_cap(rec.spec.lease_secs)) {
            return false;
        }
        self.workers.iter().any(|(id, w)| {
            self.policy.is_trusted(id)
                && w.last_engaged
                    .is_some_and(|t| now.saturating_sub(t) < VOUCHED_ACTIVE_SECS)
                && !rec.failed_by.contains(id)
                && self
                    .policy
                    .effective_capability(id, w.capability)
                    .satisfies(rec.spec.requires)
        })
    }

    /// Extend a live lease by one heartbeat window, never past its deadline.
    /// Returns the new expiry. Only the leaseholder may renew, and only while
    /// the lease is still live: a lapsed lease has already gone back to the
    /// pool, and renewing it would take it from whoever picked it up.
    pub fn renew(&mut self, worker: H160, job: &JobId, now: u64) -> Result<u64, SubmitError> {
        let rec = self.jobs.get(job).ok_or(SubmitError::UnknownJob)?;
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
        let requires = rec.spec.requires;
        if !self.tier_allowed(&worker, requires) {
            self.revoke_lease(job);
            return Err(SubmitError::TierRevoked);
        }
        let window = self.policy.lease_window_secs;
        let rec = self.jobs.get_mut(job).ok_or(SubmitError::UnknownJob)?;
        let hard = if deadline == 0 { expires_at } else { deadline };
        let extended = now.saturating_add(window).min(hard).max(expires_at);
        // Persist a heartbeat only once it has moved the expiry a full step
        // past what is on disk, or reached the deadline; a restart grants live
        // leases a fresh window anyway (see `grace_live_leases`).
        if extended
            >= rec
                .persisted_expiry
                .saturating_add(HEARTBEAT_PERSIST_STEP_SECS)
            || (extended == hard && rec.persisted_expiry != hard)
        {
            self.dirty = true;
        }
        rec.status = JobStatus::Leased {
            worker,
            expires_at: extended,
            deadline: hard,
        };
        if let Some(w) = self.workers.get_mut(&worker) {
            w.last_seen = now;
            w.last_engaged = Some(now);
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
        let rec = self.jobs.get(job).ok_or(SubmitError::UnknownJob)?;
        if let JobStatus::Leased {
            worker: holder,
            expires_at,
            ..
        } = rec.status
        {
            if holder == worker
                && expires_at > now
                && !self.tier_allowed(&worker, rec.spec.requires)
            {
                self.revoke_lease(job);
                return Err(SubmitError::TierRevoked);
            }
        }
        let trusted = self.policy.is_trusted(&worker);
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
                self.dirty = true;
                rec.result = Some(result);
                let mut group = None;
                if let Some(w) = self.workers.get_mut(&worker) {
                    w.last_seen = now;
                    w.delivered = w.delivered.saturating_add(1);
                    // Results are not verified, so only a vouched worker's
                    // delivery clears the no-show record (its own and its
                    // network's); an unvouched one leaves the back-off intact.
                    if trusted {
                        w.noshows = 0;
                        group = Some(source_group(&w.source));
                    }
                }
                if let Some(src) = group.and_then(|g| self.sources.get_mut(&g)) {
                    src.noshows = 0;
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
