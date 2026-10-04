//! `PUT /v1/fl/delta/{sha256}?job=<id>`: only the leaseholder of a federated
//! LoRA job may place a delta, only the bytes it signed, only under their own
//! content address, and only up to the round's size bound.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use citrate_training_coordinator::api::{lease_digest, router, Coordinator};
use citrate_training_coordinator::fl_upload::{fl_router, DeltaStore};
use citrate_training_coordinator::job::{Capability, JobSpec};
use citrate_training_coordinator::state::Policy;
use citrate_training_coordinator::store::Store;
use citrate_training_coordinator::JobId;
use citrate_training_worker::coordinator_protocol::{attestation_digest, fl_delta_upload_digest};
use citrate_training_worker::fl::belnap::{ConfidenceRule, WeightRule};
use citrate_training_worker::fl::delta::Artifact;
use citrate_training_worker::fl::round::{LoraDeltaPayload, RoundConfig};
use citrate_training_worker::fl::sha256;
use citrate_training_worker::wallet::Wallet;
use tower::ServiceExt;

/// Throwaway keys built at run time.
fn key(n: u64) -> String {
    format!("{:064x}", 0x2000 + n)
}

fn wallet(n: u64) -> Wallet {
    Wallet::from_hex(&key(n)).expect("wallet")
}

fn tmpdir(name: &str) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::env::temp_dir().join(format!("citrate-fl-up-{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("dir");
    d
}

fn config(roster: Vec<[u8; 20]>) -> RoundConfig {
    let mut roster = roster;
    roster.sort();
    RoundConfig {
        chain_id: 1337,
        ledger: [0x11; 20],
        cluster_id: [0x22; 32],
        base_model_sha256: [0x33; 32],
        start_adapter_sha256: [0x44; 32],
        roster,
        min_participants: 3,
        chunk_dim: 4,
        value_scale_log2: 8,
        threshold_pos: 32768,
        threshold_neg: -32768,
        confidence: ConfidenceRule::Nonzero,
        weight: WeightRule::Uniform,
        max_values: 16,
    }
}

fn probe() -> String {
    serde_json::json!({
        "schema": "nat.divergence-probe/1",
        "backend": "candle-metal", "dtype": "f32", "os": "macos", "arch": "aarch64",
        "perf": { "tokens_per_second": 13_472.0 },
        "self_repeat_identical": true,
    })
    .to_string()
}

fn nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn sig(w: &Wallet, d: &[u8; 32]) -> String {
    format!(
        "0x{}",
        hex::encode(w.sign_digest_recoverable(d).expect("sign"))
    )
}

fn post(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

struct Env {
    coord: Arc<Coordinator>,
    store: DeltaStore,
    dir: std::path::PathBuf,
    cfg: RoundConfig,
}

/// A coordinator with one federated LoRA job, leased by worker 1.
async fn leased(name: &str, cap: u64) -> Env {
    let dir = tmpdir(name);
    let w1 = wallet(1);
    let cfg = config(vec![
        w1.address().to_fixed_bytes(),
        wallet(2).address().to_fixed_bytes(),
        wallet(3).address().to_fixed_bytes(),
    ]);
    let policy = Policy {
        open_tier: Capability::Federated,
        ..Policy::default()
    };
    let coord = Arc::new(
        Coordinator::open_with(Store::new(dir.join("state.json")), policy).expect("coordinator"),
    );
    coord
        .add_job(JobSpec::new(
            "fl-0",
            Capability::Federated,
            serde_json::to_value(LoraDeltaPayload::new(cfg.clone(), 0)).expect("payload"),
        ))
        .expect("job");
    coord
        .add_job(JobSpec::new(
            "ladder-0",
            Capability::Probe,
            serde_json::json!({"task": "train"}),
        ))
        .expect("job");
    let app = router(coord.clone());
    let ts = nanos();
    let p = probe();
    let r = app
        .clone()
        .oneshot(post(
            "/v1/register",
            serde_json::json!({"probe_json": p, "timestamp": ts,
                "signature": sig(&w1, &attestation_digest(&p, ts))}),
        ))
        .await
        .expect("register");
    assert_eq!(r.status(), StatusCode::OK);
    let ts = nanos();
    let r = app
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({"timestamp": ts, "signature": sig(&w1, &lease_digest(ts))}),
        ))
        .await
        .expect("lease");
    assert_eq!(r.status(), StatusCode::OK);
    let store = DeltaStore::new(dir.join("deltas"), cap).expect("store");
    Env {
        coord,
        store,
        dir,
        cfg,
    }
}

fn artifact(cfg: &RoundConfig, worker: [u8; 20], n: usize) -> Vec<u8> {
    Artifact {
        value_scale_log2: cfg.value_scale_log2,
        round_id: cfg.round_id(0),
        worker,
        start_adapter_sha256: cfg.start_adapter_sha256,
        trained_adapter_sha256: [9; 32],
        manifest_hash: [8; 32],
        chunk_dim: cfg.chunk_dim,
        values: (0..n as i64).collect(),
    }
    .encode()
}

fn put(job: &str, sha: &[u8; 32], signer: &Wallet, body: Vec<u8>) -> Request<Body> {
    Request::builder()
        .method("PUT")
        .uri(format!("/v1/fl/delta/{}?job={job}", hex::encode(sha)))
        .header(
            "x-citrate-signature",
            sig(signer, &fl_delta_upload_digest(&JobId(job.into()), sha)),
        )
        .body(Body::from(body))
        .expect("request")
}

#[tokio::test]
async fn the_leaseholder_places_its_delta_under_its_hash() {
    let e = leased("ok", 1 << 20).await;
    let w1 = wallet(1);
    let bytes = artifact(&e.cfg, w1.address().to_fixed_bytes(), 10);
    let sha = sha256(&bytes);
    let app = fl_router(e.coord.clone(), e.store.clone());
    let r = app
        .clone()
        .oneshot(put("fl-0", &sha, &w1, bytes.clone()))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::CREATED);
    let stored = std::fs::read(e.store.path_of(&sha)).expect("stored");
    assert_eq!(stored, bytes);
    // The same bytes again are a no-op, not an error.
    let r = app
        .oneshot(put("fl-0", &sha, &w1, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::OK);
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn someone_who_does_not_hold_the_lease_is_refused() {
    let e = leased("stranger", 1 << 20).await;
    let w2 = wallet(2);
    let bytes = artifact(&e.cfg, w2.address().to_fixed_bytes(), 10);
    let sha = sha256(&bytes);
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("fl-0", &sha, &w2, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::CONFLICT);
    assert!(!e.store.path_of(&sha).exists());
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn bytes_that_do_not_hash_to_the_path_are_refused_and_not_kept() {
    let e = leased("hash", 1 << 20).await;
    let w1 = wallet(1);
    let bytes = artifact(&e.cfg, w1.address().to_fixed_bytes(), 10);
    let wrong = [7u8; 32];
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("fl-0", &wrong, &w1, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(!e.store.path_of(&wrong).exists());
    assert_eq!(
        std::fs::read_dir(e.store.dir()).expect("dir").count(),
        0,
        "no temp file is left behind"
    );
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn an_artifact_for_another_round_or_worker_is_refused() {
    let e = leased("binding", 1 << 20).await;
    let w1 = wallet(1);
    // Signed and hashed correctly, but claims to be worker 2's delta.
    let bytes = artifact(&e.cfg, wallet(2).address().to_fixed_bytes(), 10);
    let sha = sha256(&bytes);
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("fl-0", &sha, &w1, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn an_upload_over_the_round_bound_is_refused_while_streaming() {
    let e = leased("big", 1 << 20).await;
    let w1 = wallet(1);
    // max_values is 16 for this round.
    let bytes = artifact(&e.cfg, w1.address().to_fixed_bytes(), 17);
    let sha = sha256(&bytes);
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("fl-0", &sha, &w1, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn the_operator_cap_applies_below_the_round_bound() {
    let e = leased("cap", 100).await;
    let w1 = wallet(1);
    let bytes = artifact(&e.cfg, w1.address().to_fixed_bytes(), 10);
    let sha = sha256(&bytes);
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("fl-0", &sha, &w1, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn a_job_that_is_not_a_lora_round_takes_no_uploads() {
    let e = leased("task", 1 << 20).await;
    let w1 = wallet(1);
    let bytes = artifact(&e.cfg, w1.address().to_fixed_bytes(), 10);
    let sha = sha256(&bytes);
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("ladder-0", &sha, &w1, bytes.clone()))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::CONFLICT);
    let r = fl_router(e.coord.clone(), e.store.clone())
        .oneshot(put("missing", &sha, &w1, bytes))
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    let _ = std::fs::remove_dir_all(&e.dir);
}

#[tokio::test]
async fn bad_signatures_and_paths_are_refused() {
    let e = leased("sig", 1 << 20).await;
    let w1 = wallet(1);
    let bytes = artifact(&e.cfg, w1.address().to_fixed_bytes(), 10);
    let sha = sha256(&bytes);
    let app = fl_router(e.coord.clone(), e.store.clone());
    let r = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(format!("/v1/fl/delta/{}?job=fl-0", hex::encode(sha)))
                .body(Body::from(bytes.clone()))
                .expect("request"),
        )
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
    let r = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/fl/delta/not-hex?job=fl-0")
                .header("x-citrate-signature", "0x00")
                .body(Body::from(bytes))
                .expect("request"),
        )
        .await
        .expect("put");
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    let _ = std::fs::remove_dir_all(&e.dir);
}
