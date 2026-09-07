// The coordinator's operational contract.
//
// These are the properties that decide whether a volunteer fleet works or wastes
// everyone's GPU. The expensive failures are all quiet ones: a job leased twice
// produces two results and nobody notices which is authoritative; a lease that
// never expires strands work on a machine that was switched off; a submission
// accepted from a non-leaseholder lets any registered worker overwrite any
// result.

use super::*;
use crate::attestation::RegisteredWorker;
use crate::job::{Capability, JobSpec};

fn addr(b: u8) -> H160 {
    H160::repeat_byte(b)
}

fn worker(b: u8, capability: Capability) -> RegisteredWorker {
    RegisteredWorker {
        id: addr(b),
        capability,
        backend: "candle-cuda".into(),
        dtype: "bf16".into(),
        tokens_per_second: 71_098.0,
    }
}

fn job(id: &str, requires: Capability) -> JobSpec {
    JobSpec::new(id, requires, serde_json::json!({})).with_lease_secs(100)
}

fn with(jobs: &[(&str, Capability)], workers: &[(u8, Capability)]) -> State {
    let mut s = State::default();
    for (id, c) in jobs {
        s.add_job(job(id, *c));
    }
    for (b, c) in workers {
        s.register(&worker(*b, *c), 0);
    }
    s
}

// ── Assignment ─────────────────────────────────────────────────────────

#[test]
fn a_worker_is_given_a_job_it_can_do() {
    let mut s = with(&[("a", Capability::Probe)], &[(1, Capability::Probe)]);
    assert_eq!(s.lease(addr(1), 10).unwrap().id.0, "a");
}

#[test]
fn an_unregistered_worker_gets_nothing() {
    let mut s = with(&[("a", Capability::Probe)], &[]);
    assert_eq!(s.lease(addr(9), 10), Err(LeaseError::UnknownWorker));
}

/// The gate the whole capability system exists for: an underpowered machine must
/// never be handed ablation work, because its numerics would confound the result
/// rather than merely being slow.
#[test]
fn a_machine_is_never_given_work_above_its_capability() {
    let mut s = with(&[("ladder", Capability::H01)], &[(1, Capability::Federated)]);
    assert_eq!(s.lease(addr(1), 10), Err(LeaseError::NothingAvailable));
}

/// Capable machines are the scarce resource. An H-01 box taking a probe job while
/// an H-01 job waits stalls the ladder behind work any laptop could have done.
#[test]
fn a_capable_machine_takes_the_most_demanding_job_available() {
    let mut s = with(
        &[("z-probe", Capability::Probe), ("a-ladder", Capability::H01)],
        &[(1, Capability::H01)],
    );
    assert_eq!(s.lease(addr(1), 10).unwrap().id.0, "a-ladder");
}

#[test]
fn assignment_is_reproducible_across_equal_candidates() {
    for _ in 0..8 {
        let mut s = with(
            &[("b", Capability::Probe), ("a", Capability::Probe), ("c", Capability::Probe)],
            &[(1, Capability::Probe)],
        );
        assert_eq!(s.lease(addr(1), 10).unwrap().id.0, "a");
    }
}

/// The single most important safety property. Two workers on one job produce two
/// results and no rule for which is authoritative.
#[test]
fn a_job_is_never_leased_to_two_workers_at_once() {
    let mut s = with(
        &[("only", Capability::Probe)],
        &[(1, Capability::Probe), (2, Capability::Probe)],
    );
    assert_eq!(s.lease(addr(1), 10).unwrap().id.0, "only");
    assert_eq!(s.lease(addr(2), 11), Err(LeaseError::NothingAvailable));
}

// ── Expiry ─────────────────────────────────────────────────────────────

/// Volunteer machines get rebooted, thermally throttled and closed. Without this
/// the job is stranded on a machine that will never report.
#[test]
fn an_expired_lease_returns_the_job_to_the_pool() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe), (2, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    // lease_secs is 100, so at 101 it is gone.
    assert_eq!(s.expire_leases(101), vec![JobId("j".into())]);
    assert_eq!(s.lease(addr(2), 102).unwrap().id.0, "j");
}

#[test]
fn a_live_lease_is_not_expired_early() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    assert!(s.expire_leases(99).is_empty());
}

/// A job that OOMs on an 8 GB card will OOM on the same card tomorrow. Re-leasing
/// it there burns the fleet's time proving that twice.
#[test]
fn a_worker_is_not_offered_a_job_its_lease_already_expired_on() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    s.expire_leases(101);
    assert_eq!(s.lease(addr(1), 102), Err(LeaseError::NothingAvailable));
}

/// CP-B-001 (RC-8 inversion of `a_job_that_keeps_expiring_is_quarantined_...`):
/// a job that keeps being leased and expired must NOT be driven terminal.
/// Registration is unauthenticated, so terminal quarantine on no-shows is a
/// permanent, unrecoverable DoS lever; an expired lease returns the job to the
/// pool instead. A distinct worker that never failed it is still offered it.
#[test]
fn a_job_that_keeps_expiring_returns_to_the_pool_rather_than_quarantining() {
    let mut s = State::default();
    s.add_job(job("bad", Capability::Probe).with_max_attempts(2));
    for b in 1..=3u8 {
        s.register(&worker(b, Capability::Probe), 0);
    }
    let mut t = 0u64;
    for b in 1..=2u8 {
        s.lease(addr(b), t).unwrap();
        t += 101;
        s.expire_leases(t);
    }
    assert_eq!(
        s.jobs[&JobId("bad".into())].status,
        JobStatus::Pending,
        "no-shows must requeue, not permanently quarantine"
    );
    // A fresh, capable worker that never failed it IS still offered it.
    assert_eq!(s.lease(addr(3), t + 1).unwrap().id.0, "bad");
}

/// CP-B-001: registration is unauthenticated, so an attacker can lease a job
/// with a fresh throwaway key and let the lease expire, over and over. No number
/// of such no-shows may drive a job into a terminal `Quarantined` state — that
/// would be a permanent, unrecoverable denial of service on the catalogue by
/// anyone who can reach the coordinator. An expired lease returns the job to the
/// pool.
#[test]
fn distinct_fresh_keys_cannot_quarantine_a_job_by_leasing_and_expiring() {
    let mut s = State::default();
    s.add_job(job("target", Capability::Probe).with_max_attempts(3));

    let mut t = 0u64;
    for i in 0..20u64 {
        // A fresh, never-seen key each round — `failed_by` never bites. job()
        // sets lease_secs to 100, so +200 then expire is always past expiry.
        let fresh = H160::from_low_u64_be(0xF000 + i);
        s.register(&worker_probe(fresh), t);
        s.lease(fresh, t).unwrap();
        t += 200;
        s.expire_leases(t);
    }

    // The job must never become terminal, and must still be leasable.
    assert!(
        !matches!(
            s.jobs[&JobId("target".into())].status,
            JobStatus::Quarantined { .. }
        ),
        "unauthenticated no-shows must not permanently quarantine a job"
    );
    let honest = H160::from_low_u64_be(1);
    s.register(&worker_probe(honest), t + 1);
    assert_eq!(s.lease(honest, t + 1).unwrap().id.0, "target");
}

fn worker_probe(id: H160) -> RegisteredWorker {
    RegisteredWorker {
        id,
        capability: Capability::Probe,
        backend: "candle-cpu".into(),
        dtype: "f32".into(),
        tokens_per_second: 1.0,
    }
}

// ── Submission ─────────────────────────────────────────────────────────

#[test]
fn the_leaseholder_can_submit() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    assert_eq!(s.submit(addr(1), &JobId("j".into()), "result".into(), 10), Ok(()));
    assert!(matches!(s.jobs[&JobId("j".into())].status, JobStatus::Done { .. }));
}

/// A valid signature is not authorisation. Worker 2 may be perfectly honest and
/// perfectly authenticated and still have no business submitting job j.
#[test]
fn a_non_leaseholder_cannot_submit_even_when_registered() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe), (2, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    assert_eq!(
        s.submit(addr(2), &JobId("j".into()), "forged".into(), 10),
        Err(SubmitError::NotLeaseholder)
    );
}

#[test]
fn a_submission_after_the_lease_expired_is_refused() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    assert_eq!(
        s.submit(addr(1), &JobId("j".into()), "late".into(), 101),
        Err(SubmitError::LeaseExpired { expired_at: 100, now: 101 })
    );
}

#[test]
fn a_pending_job_cannot_be_submitted_against() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    assert_eq!(
        s.submit(addr(1), &JobId("j".into()), "x".into(), 1),
        Err(SubmitError::NotLeased)
    );
}

#[test]
fn submitting_an_unknown_job_is_an_error_not_a_new_job() {
    let mut s = with(&[], &[(1, Capability::Probe)]);
    assert_eq!(
        s.submit(addr(1), &JobId("ghost".into()), "x".into(), 1),
        Err(SubmitError::UnknownJob)
    );
    assert!(s.jobs.is_empty());
}

/// Re-submitting must not resurrect a finished job or overwrite its result.
#[test]
fn a_completed_job_cannot_be_submitted_against_again() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    s.submit(addr(1), &JobId("j".into()), "first".into(), 10).unwrap();
    assert_eq!(
        s.submit(addr(1), &JobId("j".into()), "second".into(), 11),
        Err(SubmitError::NotLeased)
    );
    assert_eq!(s.jobs[&JobId("j".into())].result.as_deref(), Some("first"));
}

/// Expiry must not touch finished work: a `Done` job whose old lease window has
/// passed is not a candidate for re-leasing.
#[test]
fn expiry_never_reopens_a_completed_job() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    s.submit(addr(1), &JobId("j".into()), "done".into(), 10).unwrap();
    assert!(s.expire_leases(10_000).is_empty());
    assert!(matches!(s.jobs[&JobId("j".into())].status, JobStatus::Done { .. }));
}

// ── Registration ───────────────────────────────────────────────────────

/// A member who upgrades a GPU re-probes; the new measurement must replace the
/// old rather than being ignored or duplicating the worker.
#[test]
fn re_registering_updates_the_capability_and_keeps_the_join_date() {
    let mut s = State::default();
    s.register(&worker(1, Capability::Probe), 100);
    s.register(&worker(1, Capability::H01), 500);
    assert_eq!(s.workers.len(), 1);
    let w = &s.workers[&addr(1)];
    assert_eq!(w.capability, Capability::H01);
    assert_eq!(w.registered_at, 100);
    assert_eq!(w.last_seen, 500);
}

/// CP-B-003: `/v1/register` is unauthenticated and free, and each distinct key
/// became a permanent `WorkerRecord` that was never evicted — a script could grow
/// `state.json` without bound and turn every request into an O(state) fsync. The
/// worker map must be capped: N distinct registrations leave at most
/// `MAX_WORKERS` records, not N.
#[test]
fn the_worker_map_is_bounded_regardless_of_how_many_keys_register() {
    let mut s = State::default();
    let n = MAX_WORKERS + 500;
    for i in 0..n as u64 {
        s.register(
            &RegisteredWorker {
                id: H160::from_low_u64_be(i + 1),
                capability: Capability::Probe,
                backend: "candle-cpu".into(),
                dtype: "f32".into(),
                tokens_per_second: 1.0,
            },
            i, // last_seen advances, so LRU eviction is well-defined
        );
    }
    assert!(
        s.workers.len() <= MAX_WORKERS,
        "worker map grew to {} for {n} registrations; must be capped at {MAX_WORKERS}",
        s.workers.len(),
    );
}

#[test]
fn counts_report_the_whole_fleet() {
    let mut s = with(
        &[("a", Capability::Probe), ("b", Capability::Probe)],
        &[(1, Capability::Probe)],
    );
    s.lease(addr(1), 0).unwrap();
    let c = s.counts();
    assert_eq!((c.pending, c.leased, c.done, c.workers), (1, 1, 0, 1));
}

// ── Why a forged registration buys nothing ─────────────────────────────

/// The companion to `attestation::a_key_cannot_be_made_to_vouch_for_a_better_machine`.
///
/// Editing a captured probe registers a worker id derived from the tampered bytes
/// — an identity whose private key nobody holds. This pins why that is harmless:
/// leasing requires signing as that identity, so the junk registration can never
/// be used. Registration is unprivileged on purpose (there is no roster), and the
/// safety comes from every subsequent step requiring key possession, not from
/// registration being hard.
#[test]
fn a_registration_you_cannot_sign_for_can_never_take_work() {
    let mut s = State::default();
    s.add_job(job("ladder", Capability::H01));

    // The forged identity: registered, H01, and completely unusable because the
    // lease call below is reached only by whoever can sign as `forged`.
    let forged = addr(0xEE);
    s.register(
        &RegisteredWorker {
            id: forged,
            capability: Capability::H01,
            backend: "candle-cuda".into(),
            dtype: "bf16".into(),
            tokens_per_second: 71_098.0,
        },
        0,
    );

    // An attacker holding a DIFFERENT key gets nothing from the forged entry:
    // they can only ever present their own recovered address.
    let attacker = addr(0xAA);
    assert_eq!(s.lease(attacker, 1), Err(LeaseError::UnknownWorker));

    // And the job is still there, untouched, for a machine that can prove itself.
    assert_eq!(s.jobs[&JobId("ladder".into())].status, JobStatus::Pending);
}
