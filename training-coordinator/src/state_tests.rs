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
        // A machine these tests treat as H-01 is one the operator vouched for:
        // a self-reported H-01 is only `Federated` (PBA-L3b-001).
        if *c == Capability::H01 {
            s.policy.trusted_h01.insert(addr(*b));
        }
        s.register(&worker(*b, *c), "src", 0).unwrap();
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
    // Asked once its no-show cool-down (PBA-L3b-001) is over, so what refuses it
    // here is `failed_by`, not the cool-down.
    assert_eq!(
        s.lease(addr(1), 101 + NOSHOW_BACKOFF_BASE_SECS),
        Err(LeaseError::NothingAvailable)
    );
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
        s.register(&worker(b, Capability::Probe), "src", 0).unwrap();
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
        // A distinct source each round, so the per-source cap (PBA-L3b-001)
        // does not end the loop early: this test is about quarantine.
        s.register(&worker_probe(fresh), &format!("10.0.{i}.1"), t)
            .unwrap();
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
    s.register(&worker_probe(honest), "honest", t + 1).unwrap();
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
    s.register(&worker(1, Capability::Probe), "src", 100).unwrap();
    s.register(&worker(1, Capability::H01), "src", 500).unwrap();
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
            // One key per source: this test is about the global cap.
            &format!("src-{i}"),
            i, // last_seen advances, so LRU eviction is well-defined
        )
        .unwrap();
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
        "src",
        0,
    )
    .unwrap();

    // An attacker holding a DIFFERENT key gets nothing from the forged entry:
    // they can only ever present their own recovered address.
    let attacker = addr(0xAA);
    assert_eq!(s.lease(attacker, 1), Err(LeaseError::UnknownWorker));

    // And the job is still there, untouched, for a machine that can prove itself.
    assert_eq!(s.jobs[&JobId("ladder".into())].status, JobStatus::Pending);
}

// ── PBA-L3b-001: lease squatting with throwaway keys ───────────────────
//
// Keys are free and registration is unauthenticated, so the attack is: mint a
// fresh key, claim a top-tier probe, lease the best job, never submit, and when
// the lease lapses do it again with the next key. `failed_by` never bites
// because every key is new. These tests pin each layer of the defence and,
// together, the tripwire: an honest worker gets work within a bounded number of
// cycles while a fresh-key squatter is active.

fn h01_claim(id: H160) -> RegisteredWorker {
    RegisteredWorker {
        id,
        capability: Capability::H01,
        backend: "candle-cuda".into(),
        dtype: "f32".into(),
        tokens_per_second: 1000.0,
    }
}

/// Tripwire (ported from the audit PoC `pba_l3b_lease_squat.rs`, inverted). The
/// PoC ran 50 cycles of a fresh H-01-claiming key taking the 48-hour rung the
/// moment it came free; the vouched honest worker got it 0 times. It must now get
/// it within the first cycle, because a self-reported H-01 no longer earns the
/// ladder.
#[test]
fn pba_l3b_001_honest_worker_gets_the_ladder_while_a_fresh_key_squatter_rotates() {
    let mut s = State::default();
    s.add_job(
        JobSpec::new("rung-64M", Capability::H01, serde_json::json!({})).with_lease_secs(172_800),
    );
    let honest = H160::from_low_u64_be(1);
    s.policy.trusted_h01.insert(honest);
    let mut now = 1_000u64;
    let mut first_honest_cycle = None;
    for round in 0..50u64 {
        let attacker = H160::from_low_u64_be(10_000 + round);
        let _ = s.register(&h01_claim(attacker), "203.0.113.7", now);
        // The attacker polls first, the instant anything could be free.
        let _ = s.lease(attacker, now);
        s.register(&h01_claim(honest), "198.51.100.1", now).unwrap();
        if s.lease(honest, now).is_ok() && first_honest_cycle.is_none() {
            first_honest_cycle = Some(round);
        }
        now += 172_800 + 1;
    }
    assert_eq!(
        first_honest_cycle,
        Some(0),
        "a vouched worker must win the ladder on the first cycle against a fresh-key squatter"
    );
}

/// Tripwire, below the vouched tier. For `Federated` work there is no operator
/// vetting, so the bound comes from the other layers: a lease that is not
/// heartbeated lapses after one renewal window, a lapsed key cools down and is
/// excluded from that job, and one source can hold only
/// `max_workers_per_source` identities. A squatter minting a fresh key every
/// cycle from one host runs out of identities, and the honest worker (polling
/// second every cycle) gets the job within `max_workers_per_source + 1` cycles.
#[test]
fn pba_l3b_001_fresh_key_squatter_on_one_host_is_bounded_for_federated_work() {
    let mut s = State::default();
    s.add_job(
        JobSpec::new("co-train", Capability::Federated, serde_json::json!({}))
            .with_lease_secs(172_800),
    );
    let honest = H160::from_low_u64_be(1);
    let cap = s.policy.max_workers_per_source as u64;
    let window = s.policy.lease_window_secs;
    let mut now = 1_000u64;
    let mut first_honest_cycle = None;
    for round in 0..(cap + 10) {
        let attacker = H160::from_low_u64_be(10_000 + round);
        if s.register(&h01_claim(attacker), "203.0.113.7", now).is_ok() {
            let _ = s.lease(attacker, now);
        }
        s.register(&h01_claim(honest), "198.51.100.1", now).unwrap();
        if s.lease(honest, now).is_ok() {
            first_honest_cycle = Some(round);
            break;
        }
        // The squatter never heartbeats; one window later the lease lapses.
        now += window + 1;
    }
    let got = first_honest_cycle.expect("the honest worker never obtained a lease");
    assert!(
        got <= cap,
        "honest worker waited {got} cycles; bound is {cap} (max_workers_per_source)"
    );
}

/// One key used to be able to lease every job in the queue by polling in a loop.
#[test]
fn pba_l3b_001_one_key_cannot_hold_more_than_one_lease() {
    let mut s = with(
        &[("a", Capability::Probe), ("b", Capability::Probe)],
        &[(1, Capability::Probe), (2, Capability::Probe)],
    );
    assert_eq!(s.lease(addr(1), 10).unwrap().id.0, "a");
    assert_eq!(s.lease(addr(1), 11), Err(LeaseError::AtLeaseCap));
    // The second job is still there for someone else.
    assert_eq!(s.lease(addr(2), 12).unwrap().id.0, "b");
}

/// An authenticated poll is proof of life even when there is no work. Before,
/// only a successful lease refreshed `last_seen`, so an idle honest worker was
/// the least-recently-seen entry and the first evicted by a registration flood.
#[test]
fn pba_l3b_001_an_empty_poll_refreshes_last_seen() {
    let mut s = with(&[], &[(1, Capability::Probe)]);
    assert_eq!(s.lease(addr(1), 500), Err(LeaseError::NothingAvailable));
    assert_eq!(s.workers[&addr(1)].last_seen, 500);
}

/// ...which is what keeps a polling worker alive through a flood of fresh keys.
#[test]
fn pba_l3b_001_a_polling_worker_survives_a_registration_flood() {
    let mut s = with(&[], &[(1, Capability::Probe)]);
    for i in 0..(MAX_WORKERS as u64 + 50) {
        // The honest worker polls between every few registrations.
        if i % 64 == 0 {
            let _ = s.lease(addr(1), 10 + i);
        }
        let _ = s.register(
            &worker_probe(H160::from_low_u64_be(0x10_0000 + i)),
            &format!("flood-{i}"),
            10 + i,
        );
    }
    assert!(
        s.workers.contains_key(&addr(1)),
        "a worker that keeps polling must not be evicted"
    );
}

/// A key that leases and walks away cannot immediately lease the next job.
#[test]
fn pba_l3b_001_a_lapsed_lease_starts_a_doubling_cool_down() {
    let mut s = with(
        &[("a", Capability::Probe), ("b", Capability::Probe), ("c", Capability::Probe)],
        &[(1, Capability::Probe)],
    );
    s.lease(addr(1), 0).unwrap(); // "a", lapses at 100
    let first = 101 + NOSHOW_BACKOFF_BASE_SECS;
    assert_eq!(
        s.lease(addr(1), 101),
        Err(LeaseError::CoolingDown { until: first })
    );
    assert_eq!(s.lease(addr(1), first - 1), Err(LeaseError::CoolingDown { until: first }));
    // Cool-down over: it may lease again (not "a", which it already failed).
    assert_eq!(s.lease(addr(1), first).unwrap().id.0, "b");
    // A second consecutive no-show doubles the wait.
    let lapse = first + 100;
    assert_eq!(
        s.lease(addr(1), lapse),
        Err(LeaseError::CoolingDown {
            until: lapse + 2 * NOSHOW_BACKOFF_BASE_SECS
        })
    );
}

/// A delivered result clears the no-show count, so an honest machine that once
/// lost power is not penalised forever.
#[test]
fn pba_l3b_001_a_submission_clears_the_no_show_count() {
    let mut s = with(
        &[("a", Capability::Probe), ("b", Capability::Probe), ("c", Capability::Probe)],
        &[(1, Capability::Probe)],
    );
    s.lease(addr(1), 0).unwrap();
    let t = 101 + NOSHOW_BACKOFF_BASE_SECS;
    assert!(s.lease(addr(1), 101).is_err());
    let b = s.lease(addr(1), t).unwrap();
    s.submit(addr(1), &b.id, "ok".into(), t + 1).unwrap();
    assert_eq!(s.workers[&addr(1)].noshows, 0);
}

#[test]
fn pba_l3b_001_noshow_backoff_doubles_and_is_capped() {
    assert_eq!(noshow_backoff(1), NOSHOW_BACKOFF_BASE_SECS);
    assert_eq!(noshow_backoff(2), 2 * NOSHOW_BACKOFF_BASE_SECS);
    assert_eq!(noshow_backoff(3), 4 * NOSHOW_BACKOFF_BASE_SECS);
    assert_eq!(noshow_backoff(40), NOSHOW_BACKOFF_MAX_SECS);
    assert_eq!(noshow_backoff(u32::MAX), NOSHOW_BACKOFF_MAX_SECS);
}

/// A lease lives one renewal window unless heartbeated, not the job's whole
/// `lease_secs` (which is 50 hours for the big rungs).
#[test]
fn pba_l3b_001_an_unrenewed_lease_lapses_after_one_window() {
    let mut s = State::default();
    s.add_job(JobSpec::new("long", Capability::Probe, serde_json::json!({})).with_lease_secs(172_800));
    s.register(&worker_probe(addr(1)), "src", 0).unwrap();
    s.lease(addr(1), 0).unwrap();
    let window = s.policy.lease_window_secs;
    assert!(s.expire_leases(window - 1).is_empty());
    assert_eq!(s.expire_leases(window), vec![JobId("long".into())]);
}

/// Heartbeats extend a lease one window at a time, up to the job's deadline and
/// never past it; after the lease lapses a heartbeat cannot resurrect it.
#[test]
fn pba_l3b_001_heartbeats_extend_up_to_the_deadline_and_no_further() {
    let mut s = State::default();
    s.policy.lease_window_secs = 900;
    s.add_job(JobSpec::new("j", Capability::Probe, serde_json::json!({})).with_lease_secs(2_000));
    s.register(&worker_probe(addr(1)), "src", 0).unwrap();
    s.lease(addr(1), 0).unwrap();
    let j = JobId("j".into());
    assert_eq!(s.renew(addr(1), &j, 800), Ok(1_700));
    assert_eq!(s.renew(addr(1), &j, 1_600), Ok(2_000));
    assert_eq!(s.renew(addr(1), &j, 1_990), Ok(2_000));
    assert_eq!(
        s.renew(addr(1), &j, 2_000),
        Err(SubmitError::LeaseExpired { expired_at: 2_000, now: 2_000 })
    );
    // A renewal that lands before the lease's current expiry never shortens it.
    let mut s2 = State::default();
    s2.policy.lease_window_secs = 10;
    s2.add_job(JobSpec::new("j", Capability::Probe, serde_json::json!({})).with_lease_secs(100));
    s2.register(&worker_probe(addr(1)), "src", 0).unwrap();
    s2.lease(addr(1), 0).unwrap();
    assert_eq!(s2.renew(addr(1), &j, 5), Ok(15));
    assert_eq!(s2.renew(addr(1), &j, 1), Ok(15));
}

#[test]
fn pba_l3b_001_only_the_leaseholder_can_renew() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe), (2, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    let j = JobId("j".into());
    assert_eq!(s.renew(addr(2), &j, 10), Err(SubmitError::NotLeaseholder));
    assert_eq!(s.renew(addr(1), &JobId("ghost".into()), 10), Err(SubmitError::UnknownJob));
    let mut p = with(&[("p", Capability::Probe)], &[(1, Capability::Probe)]);
    assert_eq!(p.renew(addr(1), &JobId("p".into()), 10), Err(SubmitError::NotLeased));
}

/// A lease record written before `deadline` existed (deadline == 0) can be
/// renewed only up to its recorded expiry, never indefinitely.
#[test]
fn pba_l3b_001_a_legacy_lease_record_is_not_extended() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    let j = JobId("j".into());
    s.jobs.get_mut(&j).unwrap().status = JobStatus::Leased {
        worker: addr(1),
        expires_at: 50,
        deadline: 0,
    };
    assert_eq!(s.renew(addr(1), &j, 10), Ok(50));
}

/// The probe is written by the machine itself, so a self-reported H-01 is a
/// claim. Only a vouched address gets the ladder.
#[test]
fn pba_l3b_001_a_self_reported_h01_is_not_trusted_for_the_ladder() {
    let mut s = State::default();
    s.add_job(job("ladder", Capability::H01));
    s.add_job(job("co-train", Capability::Federated));
    let stranger = addr(0xAA);
    assert_eq!(
        s.register(&h01_claim(stranger), "a", 0),
        Ok(Capability::Federated)
    );
    assert_eq!(s.lease(stranger, 1).unwrap().id.0, "co-train");

    let vouched = addr(0xBB);
    s.policy.trusted_h01.insert(vouched);
    assert_eq!(s.register(&h01_claim(vouched), "b", 0), Ok(Capability::H01));
    assert_eq!(s.lease(vouched, 1).unwrap().id.0, "ladder");
    // Vouching does not upgrade a lesser claim.
    assert_eq!(
        s.policy.effective_capability(&vouched, Capability::Probe),
        Capability::Probe
    );
}

/// Removing an address from the vouched list takes effect at the next lease,
/// even though its stored registration still says H-01.
#[test]
fn pba_l3b_001_the_trust_decision_is_made_at_lease_time() {
    let mut s = with(&[("ladder", Capability::H01)], &[(1, Capability::H01)]);
    s.policy.trusted_h01.clear();
    assert_eq!(s.lease(addr(1), 1), Err(LeaseError::NothingAvailable));
}

#[test]
fn pba_l3b_001_one_source_cannot_register_unbounded_identities() {
    let mut s = State::default();
    let cap = s.policy.max_workers_per_source as u64;
    for i in 0..cap {
        s.register(&worker_probe(H160::from_low_u64_be(100 + i)), "a", 0)
            .unwrap();
    }
    let extra = worker_probe(H160::from_low_u64_be(999));
    assert_eq!(s.register(&extra, "a", 1), Err(RegisterError::SourceFull));
    // A refresh from a full source is always fine.
    assert_eq!(
        s.register(&worker_probe(H160::from_low_u64_be(100)), "a", 2),
        Ok(Capability::Probe)
    );
    // Another source is unaffected.
    assert!(s.register(&extra, "b", 3).is_ok());
    // A vouched machine is never capped by its source.
    let lab = H160::from_low_u64_be(4242);
    s.policy.trusted_h01.insert(lab);
    assert!(s.register(&worker_probe(lab), "a", 4).is_ok());
}

/// A slot whose worker has gone quiet for a day, and holds no lease, is
/// reclaimed, so a member who re-keys is not locked out forever.
#[test]
fn pba_l3b_001_a_stale_source_slot_is_reclaimed() {
    let mut s = State::default();
    s.policy.max_workers_per_source = 2;
    s.add_job(job("j", Capability::Probe).with_lease_secs(10 * SOURCE_SLOT_STALE_SECS));
    s.policy.lease_window_secs = 10 * SOURCE_SLOT_STALE_SECS;
    s.register(&worker_probe(addr(1)), "a", 0).unwrap();
    s.register(&worker_probe(addr(2)), "a", 10).unwrap();
    // addr(1) is mid-job: its slot is not reclaimable even when stale.
    s.lease(addr(1), 10).unwrap();
    let later = 10 + SOURCE_SLOT_STALE_SECS;
    assert_eq!(
        s.register(&worker_probe(addr(3)), "a", later - 1),
        Err(RegisterError::SourceFull)
    );
    s.register(&worker_probe(addr(3)), "a", later).unwrap();
    assert!(s.workers.contains_key(&addr(1)), "a leaseholder keeps its slot");
    assert!(!s.workers.contains_key(&addr(2)), "the stale idle slot is reclaimed");
}

/// The global cap never evicts a vouched machine or one that is mid-job; when
/// only such workers remain, a new registration is refused instead.
#[test]
fn pba_l3b_001_the_global_cap_never_evicts_a_vouched_or_busy_worker() {
    let mut s = State::default();
    for i in 0..MAX_WORKERS as u64 {
        let id = H160::from_low_u64_be(i + 1);
        s.policy.trusted_h01.insert(id);
        s.register(&worker_probe(id), &format!("s{i}"), i).unwrap();
    }
    assert_eq!(
        s.register(&worker_probe(addr(0xEE)), "x", 99_999),
        Err(RegisterError::RegistryFull)
    );
    // Make the oldest one untrusted: it becomes the eviction victim.
    s.policy.trusted_h01.remove(&H160::from_low_u64_be(1));
    s.register(&worker_probe(addr(0xEE)), "x", 99_999).unwrap();
    assert!(!s.workers.contains_key(&H160::from_low_u64_be(1)));

    // A busy (leaseholding) untrusted worker is skipped in favour of an idle one.
    let mut b = State::default();
    b.add_job(job("j", Capability::Probe));
    for i in 0..MAX_WORKERS as u64 {
        b.register(&worker_probe(H160::from_low_u64_be(i + 1)), &format!("s{i}"), i)
            .unwrap();
    }
    let oldest = H160::from_low_u64_be(1);
    b.lease(oldest, 0).unwrap(); // does not refresh ordering past others: last_seen = 0
    b.register(&worker_probe(addr(0xEE)), "x", 50).unwrap();
    assert!(b.workers.contains_key(&oldest), "the leaseholder is not evicted");
    assert!(!b.workers.contains_key(&H160::from_low_u64_be(2)));
}

#[test]
fn pba_l3b_001_the_vouched_list_parses_strictly() {
    let a = "0x00000000000000000000000000000000000000aa";
    let b = "00000000000000000000000000000000000000Bb";
    let set = parse_address_list(&format!(" {a}, ,{b} ")).unwrap();
    assert_eq!(set.len(), 2);
    assert!(set.contains(&H160::from_low_u64_be(0xaa)));
    assert!(set.contains(&H160::from_low_u64_be(0xbb)));
    assert!(parse_address_list("").unwrap().is_empty());
    assert!(parse_address_list("0x1234").is_err());
    assert!(parse_address_list("0xzz00000000000000000000000000000000000000").is_err());
}

/// Kills cargo-mutants survivors: a lease that has already lapsed (expiry ==
/// now) does not protect its holder's source slot, and `attempts` counts leases.
#[test]
fn pba_l3b_001_a_lapsed_lease_does_not_protect_a_source_slot() {
    let mut s = State::default();
    s.policy.max_workers_per_source = 1;
    s.add_job(job("j", Capability::Probe));
    s.register(&worker_probe(addr(1)), "a", 0).unwrap();
    let later = SOURCE_SLOT_STALE_SECS;
    s.jobs.get_mut(&JobId("j".into())).unwrap().status = JobStatus::Leased {
        worker: addr(1),
        expires_at: later,
        deadline: later,
    };
    s.register(&worker_probe(addr(2)), "a", later).unwrap();
    assert!(!s.workers.contains_key(&addr(1)));
}

#[test]
fn pba_l3b_001_each_lease_counts_one_attempt() {
    let mut s = with(&[("j", Capability::Probe)], &[(1, Capability::Probe)]);
    s.lease(addr(1), 0).unwrap();
    assert_eq!(s.jobs[&JobId("j".into())].attempts, 1);
}
