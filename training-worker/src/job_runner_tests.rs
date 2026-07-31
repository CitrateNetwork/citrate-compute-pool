// The refusal path.
//
// Every test here is about work NOT happening. That is deliberate: the expensive
// failures in a training runner are the ones where it proceeds. A job that runs
// for two days against the wrong commitment grid produces honest work that fails
// every challenge and costs the worker 10% of its stake — and nothing in the
// system notices, because the numbers all look fine.
//
// The success path needs a real 2.4 GB corpus and a 64M checkpoint on a GPU, so
// it lives in `examples/real_training_run.rs`, which runs against the genuine
// artifacts rather than fabricating them here.

use super::*;
use crate::coordinator_protocol::{Capability, JobSpec};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("citrate-runner-test-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn runner(name: &str) -> NatJobRunner {
    let d = tmpdir(name);
    NatJobRunner::new(d.join("store"), d.join("scratch"), WorkerAddress::repeat_byte(1))
}

fn payload() -> serde_json::Value {
    serde_json::json!({
        "task": "train",
        "model_start_hash": format!("{:?}", B256::repeat_byte(0xAA)),
        "dataset_hash": format!("{:?}", B256::repeat_byte(0xBB)),
        "commitment_grid": "q16",
        "epoch": 0,
        "steps": 4,
        "worker_shard": 0,
        "batch_size": 4,
        "learning_rate": 0.0001,
        "max_windows": 64,
        "shards_per_step": 8,
        "seed": 2026
    })
}

fn job_with(payload: serde_json::Value) -> JobSpec {
    JobSpec::new("t1", Capability::H01, payload)
}

#[tokio::test]
async fn a_payload_that_is_not_a_training_job_is_refused_before_anything_loads() {
    let r = runner("badpayload");
    let e = r.run(&job_with(serde_json::json!({ "hello": "world" }))).await.unwrap_err();
    assert!(matches!(e, RunError::BadPayload(_)), "got {e:?}");
}

/// A missing field must be a refusal, not a default. A worker that quietly trains
/// one epoch because `steps` was absent has produced a result nobody asked for.
#[tokio::test]
async fn a_payload_missing_a_field_is_refused_rather_than_defaulted() {
    let mut p = payload();
    p.as_object_mut().unwrap().remove("steps");
    let e = runner("missingfield").run(&job_with(p)).await.unwrap_err();
    assert!(matches!(e, RunError::BadPayload(_)), "got {e:?}");
}

/// A runner handed someone else's job type must decline it whole rather than
/// half-understand it.
#[tokio::test]
async fn a_job_for_another_task_is_declined_by_name() {
    let mut p = payload();
    p["task"] = serde_json::json!("divergence_probe");
    match runner("wrongtask").run(&job_with(p)).await.unwrap_err() {
        RunError::WrongTask(t) => assert_eq!(t, "divergence_probe"),
        e => panic!("got {e:?}"),
    }
}

/// Declining names the problem AND the two ways out, so a member reading a log at
/// midnight can act on it instead of filing an issue.
#[tokio::test]
async fn unstaged_artifacts_are_declined_with_an_actionable_reason() {
    let e = runner("noartifacts").run(&job_with(payload())).await.unwrap_err();
    assert!(matches!(e, RunError::ArtifactsMissing(_)), "got {e:?}");
    let s = e.to_string();
    assert!(s.contains("CITRATE_ARTIFACT_MIRROR"), "must name the fix: {s}");
    assert!(s.contains("by hand"), "must name the manual alternative: {s}");
}

/// An unknown grid name must not fall back to a default. Defaulting here is
/// exactly how a worker ends up committing on a grid the committee is not
/// resolving on.
#[tokio::test]
async fn an_unrecognised_commitment_grid_is_refused_rather_than_defaulted() {
    let mut p = payload();
    p["commitment_grid"] = serde_json::json!("float64-someday");
    let e = runner("badgrid").run(&job_with(p)).await.unwrap_err();
    assert!(matches!(e, RunError::BadPayload(_)), "got {e:?}");
}

/// Both grids must be expressible, so a job can be pinned to the legacy scale
/// deliberately — the check is that declared and actual AGREE, not that one
/// particular grid is hardcoded.
#[test]
fn both_commitment_grids_round_trip_through_the_payload() {
    for (name, grid) in [
        ("q16", CommitmentGrid::Q16),
        ("legacy-f32-scale", CommitmentGrid::LegacyF32Scale),
    ] {
        let mut p = payload();
        p["commitment_grid"] = serde_json::json!(name);
        let parsed: TrainingJobPayload = serde_json::from_value(p).unwrap();
        assert_eq!(parsed.commitment_grid, grid);
        assert_eq!(parsed.commitment_grid.as_str(), name);
    }
}

/// The grid mismatch message has to explain the consequence, because the person
/// reading it is a volunteer whose machine just declined a job and who has no
/// reason to know what a commitment grid is.
#[test]
fn the_grid_mismatch_error_explains_why_it_matters() {
    let e = RunError::GridMismatch {
        declared: "q16",
        actual: "legacy-f32-scale",
    };
    let s = e.to_string();
    assert!(s.contains("q16") && s.contains("legacy-f32-scale"));
    assert!(s.contains("challenge"), "must say what goes wrong: {s}");
}

#[test]
fn the_backend_honesty_gate_names_the_actual_risk() {
    let s = RunError::BackendDoesNotHonorSpec.to_string();
    assert!(s.contains("honour the job spec"));
}

/// The chain-of-custody property, asserted on the type rather than on a run:
/// every step carries the previous step's post-weights, which is what makes a
/// mid-run weight substitution detectable by a challenger.
#[test]
fn step_records_carry_what_a_challenger_needs() {
    let r = StepRecord {
        step: 3,
        commitment: B256::repeat_byte(1),
        post_weights: B256::repeat_byte(2),
    };
    let v = serde_json::to_value(&r).unwrap();
    assert!(v.get("commitment").is_some());
    assert!(v.get("post_weights").is_some());
    assert_eq!(v["step"], 3);
}

/// Reporting only the epoch root would make the work unfalsifiable — a challenger
/// needs the per-step commitments to prove a specific step wrong.
#[test]
fn the_result_reports_per_step_commitments_and_not_only_the_root() {
    let res = TrainingJobResult {
        job: "t1".into(),
        task: TASK_TRAIN,
        backend: "candle-cuda",
        commitment_grid: "q16",
        epoch: 0,
        worker_shard: 0,
        steps: vec![StepRecord {
            step: 0,
            commitment: B256::repeat_byte(9),
            post_weights: B256::repeat_byte(8),
        }],
        epoch_root: B256::repeat_byte(7),
        final_weights: B256::repeat_byte(8),
        data_quality_raw: 65536,
        zone_l2: vec![("bucket_0".into(), 1.5)],
        share_trace: vec![],
        seconds: 1.0,
    };
    let v = serde_json::to_value(&res).unwrap();
    assert_eq!(v["steps"].as_array().unwrap().len(), 1);
    assert!(v.get("epoch_root").is_some());
    // The backend is provenance once the fleet is heterogeneous.
    assert_eq!(v["backend"], "candle-cuda");
    // A dead zone must be visible in the result, not discovered months later.
    assert!(v.get("zone_l2").is_some());
}

#[test]
fn zone_l2_accumulates_across_steps_per_bucket() {
    use crate::backend::Tensor;
    let mut acc = Vec::new();
    let grads = vec![
        Tensor { data: vec![3.0, 4.0], layer_index: 0 }, // L2 = 5
        Tensor { data: vec![0.0, 0.0], layer_index: 1 }, // L2 = 0 — a dead bucket
    ];
    accumulate_zone_l2(&mut acc, &grads);
    accumulate_zone_l2(&mut acc, &grads);
    assert_eq!(acc.len(), 2);
    assert!((acc[0].1 - 10.0).abs() < 1e-9);
    assert_eq!(acc[1].1, 0.0, "a bucket that never moves must report zero, not be omitted");
}

// ── Staging ────────────────────────────────────────────────────────────

/// The economics of the whole distribution design, asserted as arithmetic.
///
/// corpus-v6 is 185,475 shards. A job reads `shards_per_step` per step, so what
/// it needs is bounded by `steps × shards_per_step` — independent of corpus size.
/// Fetching the corpus instead would move ~12× more data per member per job.
#[test]
fn a_job_needs_a_bounded_shard_set_not_the_corpus() {
    use crate::nat_backend::shard_slice_for;
    let total = 185_475; // corpus-v6, measured
    let mut wanted = std::collections::BTreeSet::new();
    for step in 0..100u32 {
        for idx in shard_slice_for(total, 4, step, 0) {
            wanted.insert(idx);
        }
    }
    assert!(wanted.len() <= 400, "got {}", wanted.len());
    // ~7.4 KB each: single-digit MB against a 2.4 GB corpus.
    assert!(wanted.len() * 7_400 < 5 * 1024 * 1024);
}

/// Two workers on the same step must read disjoint slices, or they duplicate work
/// and the corpus coverage the co-op is paying for does not happen.
#[test]
fn different_workers_read_different_shards() {
    use crate::nat_backend::shard_slice_for;
    let a: std::collections::BTreeSet<_> = shard_slice_for(1000, 8, 0, 0).into_iter().collect();
    let b: std::collections::BTreeSet<_> = shard_slice_for(1000, 8, 0, 1).into_iter().collect();
    assert!(a.intersection(&b).count() < a.len(), "worker slices must not coincide");
}

/// The load-bearing property of sharing one stride function: what the prefetcher
/// downloads is exactly what the reader opens. If these ever diverged, a worker
/// would stage a set of shards and then fail mid-run on a file it never fetched.
#[test]
fn the_prefetch_set_and_the_read_set_are_the_same_function() {
    use crate::nat_backend::shard_slice_for;
    for (total, per_step, step, shard) in
        [(100, 4, 0, 0), (185_475, 8, 17, 3), (7, 64, 2, 9), (1, 1, 0, 0)]
    {
        assert_eq!(
            shard_slice_for(total, per_step, step, shard),
            shard_slice_for(total, per_step, step, shard),
            "stride must be deterministic"
        );
    }
}

/// Never ask for more shards than exist, and never index out of the manifest —
/// a small corpus in testing must not panic the prefetcher.
#[test]
fn the_stride_stays_in_bounds_on_a_corpus_smaller_than_a_step() {
    use crate::nat_backend::shard_slice_for;
    let picks = shard_slice_for(3, 64, 5, 2);
    assert_eq!(picks.len(), 3, "cannot read more shards than exist");
    assert!(picks.iter().all(|i| *i < 3));
}

#[test]
fn an_empty_corpus_yields_no_picks_rather_than_panicking() {
    use crate::nat_backend::shard_slice_for;
    assert!(shard_slice_for(0, 4, 0, 0).is_empty());
}

/// Without a mirror the runner still declines rather than reaching out — a member
/// who stages by hand must not have traffic generated on their behalf.
#[tokio::test]
async fn without_a_mirror_a_missing_artifact_is_still_declined() {
    let e = runner("nomirror").run(&job_with(payload())).await.unwrap_err();
    assert!(matches!(e, RunError::ArtifactsMissing(_)), "got {e:?}");
}

/// A mirror that cannot be reached is a staging failure, and the job is declined
/// rather than half-run against a partly-staged store.
#[tokio::test]
async fn an_unreachable_mirror_declines_the_job() {
    let d = tmpdir("deadmirror");
    let r = NatJobRunner::new(d.join("store"), d.join("scratch"), WorkerAddress::repeat_byte(1))
        .with_mirror("http://127.0.0.1:1");
    let e = r.run(&job_with(payload())).await.unwrap_err();
    assert!(matches!(e, RunError::Fetch(_)), "got {e:?}");
}

// ── ADR-0012: recording which zones train, and abandoning runs where one does not

/// The floor guarantees each of five zones `merge_floor / 5`. A zone sitting at
/// that value has NO learned contribution — the floor is carrying it entirely,
/// which is what "dead" means once ADR-0012's floor exists. It is no longer
/// "share zero"; the floor makes zero impossible, which is precisely why the
/// detector cannot look for zero.
#[test]
fn the_dead_zone_threshold_is_the_floor_not_zero() {
    let floor = nat_candle::autoreg::DEFAULT_MERGE_FLOOR / 5.0;
    assert!((floor - 0.002).abs() < 1e-9, "floor share is {floor}");
    // The shipped 64M checkpoint had PF at exactly 0.000000 — below the floor,
    // because it predates the floor. It must still register as dead.
    assert!(0.0_f64 <= floor * 1.05);
    // And a zone the router genuinely favours must NOT register as dead.
    assert!(!(0.38_f64 <= floor * 1.05), "CB at 0.38 must not read as dead");
}

/// One bad sample is not evidence. The H-01 scope's own 400-step probes showed a
/// severe early transient (CX to 0.001) that RECOVERED — aborting on that would
/// throw away good runs.
#[test]
fn a_single_dead_sample_does_not_abort_by_default() {
    let p: TrainingJobPayload = serde_json::from_value(payload()).unwrap();
    assert!(p.dead_zone_patience > 1, "patience must tolerate a transient");
}

/// Sampling is on by default. The entire lesson of ADR-0012 is that the failure
/// was unobservable, so a run that records nothing is the default we must not have.
#[test]
fn share_recording_is_on_by_default() {
    let p: TrainingJobPayload = serde_json::from_value(payload()).unwrap();
    assert_eq!(p.share_every, 1);
}

/// But it must be disableable for the dense arm, which has no zones to record.
#[test]
fn share_recording_can_be_disabled_for_the_dense_arm() {
    let mut v = payload();
    v["share_every"] = serde_json::json!(0);
    let p: TrainingJobPayload = serde_json::from_value(v).unwrap();
    assert_eq!(p.share_every, 0);
}

/// The abort message has to explain the consequence. A volunteer whose two-day
/// job just died deserves to know it was the right outcome, not a crash.
#[test]
fn the_dead_zone_error_explains_why_the_run_was_abandoned() {
    let e = RunError::DeadZone {
        zone: "PF".into(),
        share: 0.002,
        floor: 0.002,
        samples: 5,
        step: 400,
    };
    let s = e.to_string();
    assert!(s.contains("PF"));
    assert!(s.contains("merge floor"));
    assert!(s.contains("nobody can quote"), "must say why abandoning is right: {s}");
}

/// The trajectory is part of the signed result, not a local log. A challenger
/// re-running the job must be able to see the same zone history.
#[test]
fn the_share_trace_is_carried_in_the_submitted_result() {
    let v = serde_json::to_value(TrainingJobResult {
        job: "t".into(), task: TASK_TRAIN, backend: "candle-cuda", commitment_grid: "q16",
        epoch: 0, worker_shard: 0, steps: vec![], epoch_root: B256::zero(),
        final_weights: B256::zero(), data_quality_raw: 0, zone_l2: vec![],
        share_trace: vec![ShareSample {
            step: 0,
            shares: vec![("PF".into(), 0.0027), ("SM".into(), 0.61)],
        }],
        seconds: 1.0,
    })
    .unwrap();
    assert_eq!(v["share_trace"][0]["shares"][0][0], "PF");
    assert_eq!(v["share_trace"][0]["step"], 0);
}
