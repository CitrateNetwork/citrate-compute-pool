//! The whole loop over the real router: register → lease → submit → status.
//!
//! The state machine is unit-tested in `state.rs`. This exists because the
//! interesting failures of a service are in the wiring — a route that recovers
//! the signer from the wrong bytes, or an error mapped to a status code that
//! makes a worker retry forever. None of those are visible to a unit test.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use citrate_training_coordinator::api::{lease_digest, router, Coordinator};
use citrate_training_coordinator::job::{Capability, JobSpec};
use citrate_training_coordinator::store::Store;
use citrate_training_coordinator::submission::submission_digest;
use citrate_training_coordinator::JobId;
use citrate_training_worker::wallet::Wallet;
use http_body_util::BodyExt;
use sha3::{Digest, Keccak256};
use tower::ServiceExt;

// Anvil account #0 / #1 — well-known throwaways, NOT real keys.
const KEY_A: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const KEY_B: &str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("citrate-coord-http-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.join("state.json")
}

fn probe(backend: &str, dtype: &str, tok_s: f64, self_repeat: bool) -> String {
    serde_json::json!({
        "schema": "nat.divergence-probe/1",
        "backend": backend, "dtype": dtype, "os": "linux", "arch": "x86_64",
        "perf": { "tokens_per_second": tok_s },
        "self_repeat_identical": self_repeat,
    })
    .to_string()
}

fn hex_sig(key: &str, digest: &[u8; 32]) -> String {
    let w = Wallet::from_hex(key).unwrap();
    format!(
        "0x{}",
        hex::encode(w.sign_digest_recoverable(digest).unwrap())
    )
}

fn keccak(b: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(b);
    let mut d = [0u8; 32];
    d.copy_from_slice(&h.finalize());
    d
}

fn post(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn json_of(res: axum::response::Response) -> serde_json::Value {
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

fn coordinator(name: &str, jobs: Vec<JobSpec>) -> Arc<Coordinator> {
    let c = Arc::new(Coordinator::open(Store::new(tmp(name))).unwrap());
    for j in jobs {
        c.add_job(j).unwrap();
    }
    c
}

fn register_body(key: &str, body: &str) -> serde_json::Value {
    serde_json::json!({
        "probe_json": body,
        "signature": hex_sig(key, &keccak(body.as_bytes())),
    })
}

#[tokio::test]
async fn a_worker_registers_leases_submits_and_the_status_reflects_it() {
    let c = coordinator(
        "happy",
        vec![JobSpec::new(
            "h01-64m-seed2",
            Capability::H01,
            serde_json::json!({"rung":"64M"}),
        )],
    );

    // register
    let res = router(c.clone())
        .oneshot(post(
            "/v1/register",
            register_body(KEY_A, &probe("candle-cuda", "f32", 71_098.0, true)),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(json_of(res).await["capability"], "h01");

    // lease
    let res = router(c.clone())
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({ "signature": hex_sig(KEY_A, &lease_digest()) }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let job = json_of(res).await;
    assert_eq!(job["id"], "h01-64m-seed2");

    // submit
    let payload = r#"{"loss":2.31}"#;
    let digest = submission_digest(&JobId("h01-64m-seed2".into()), payload);
    let res = router(c.clone())
        .oneshot(post(
            "/v1/submit",
            serde_json::json!({ "job": "h01-64m-seed2", "payload": payload, "signature": hex_sig(KEY_A, &digest) }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::ACCEPTED);

    // status
    let res = router(c.clone())
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let s = json_of(res).await;
    assert_eq!(s["counts"]["done"], 1);
    assert_eq!(s["counts"]["workers"], 1);
    // Nothing is paid until the settlement tolerance is set from fleet data.
    assert_eq!(s["settlement"], "shadow");
}

/// A CPU machine must not be handed ablation work over the wire either — the
/// capability gate has to live behind the route, not only in the state machine.
#[tokio::test]
async fn a_cpu_machine_is_told_there_is_no_work_rather_than_given_the_ladder() {
    let c = coordinator(
        "gated",
        vec![JobSpec::new(
            "ladder",
            Capability::H01,
            serde_json::json!({}),
        )],
    );

    let res = router(c.clone())
        .oneshot(post(
            "/v1/register",
            register_body(KEY_A, &probe("candle-cpu", "f32", 7_786.0, true)),
        ))
        .await
        .unwrap();
    assert_eq!(json_of(res).await["capability"], "probe");

    let res = router(c.clone())
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({ "signature": hex_sig(KEY_A, &lease_digest()) }),
        ))
        .await
        .unwrap();
    // 204, not an error: having nothing to do is the steady state of a fleet
    // with more machines than queued work, and workers poll on it.
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
}

/// The authorisation boundary, over HTTP. B's signature is perfectly valid and
/// B is perfectly registered; B still may not submit A's job.
#[tokio::test]
async fn a_valid_signature_from_the_wrong_worker_is_refused() {
    let c = coordinator(
        "wrongworker",
        vec![JobSpec::new("j", Capability::Probe, serde_json::json!({}))],
    );
    let good = probe("candle-cuda", "f32", 71_098.0, true);

    for k in [KEY_A, KEY_B] {
        let res = router(c.clone())
            .oneshot(post("/v1/register", register_body(k, &good)))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
    // A takes the job.
    let res = router(c.clone())
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({ "signature": hex_sig(KEY_A, &lease_digest()) }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // B submits it, correctly signed as B.
    let payload = "stolen";
    let digest = submission_digest(&JobId("j".into()), payload);
    let res = router(c.clone())
        .oneshot(post(
            "/v1/submit",
            serde_json::json!({ "job": "j", "payload": payload, "signature": hex_sig(KEY_B, &digest) }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn a_garbage_signature_is_unauthorized_not_a_panic() {
    let c = coordinator("garbage", vec![]);
    let res = router(c)
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({ "signature": "0x00" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

/// The reason the store is crash-atomic. A restart that forgot its leases would
/// hand the same job to a second machine while the first is still training.
#[tokio::test]
async fn state_survives_a_restart_and_the_lease_is_still_held() {
    let path = tmp("restart");
    let first = Arc::new(Coordinator::open(Store::new(&path)).unwrap());
    first
        .add_job(JobSpec::new("j", Capability::Probe, serde_json::json!({})).with_lease_secs(9999))
        .unwrap();
    let good = probe("candle-cuda", "f32", 71_098.0, true);
    router(first.clone())
        .oneshot(post("/v1/register", register_body(KEY_A, &good)))
        .await
        .unwrap();
    let res = router(first.clone())
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({ "signature": hex_sig(KEY_A, &lease_digest()) }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    drop(first);

    // A brand-new process over the same file.
    let second = Arc::new(Coordinator::open(Store::new(&path)).unwrap());
    let counts = second.snapshot().counts();
    assert_eq!((counts.leased, counts.pending, counts.workers), (1, 0, 1));

    // And a different machine is not handed the same job.
    router(second.clone())
        .oneshot(post("/v1/register", register_body(KEY_B, &good)))
        .await
        .unwrap();
    let res = router(second.clone())
        .oneshot(post(
            "/v1/lease",
            serde_json::json!({ "signature": hex_sig(KEY_B, &lease_digest()) }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
}

// ── The real client against the real server ────────────────────────────
//
// Everything above drives the router with hand-built requests. These drive it
// with the actual `CoordinatorClient` a member's machine runs, over a real
// socket. This crate is the only one that may depend on both sides, so this is
// the only place the seam can be tested at all — and the seam is exactly where a
// digest mismatch would live: two implementations of the same signing preimage
// agree until one changes a separator, and then every honest submission fails to
// authenticate while looking like a key problem.

use citrate_training_worker::coordinator_client::{Backoff, CoordinatorClient};

/// Serve the real router on an ephemeral port and hand back its base URL.
async fn serve(c: Arc<Coordinator>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router(c)).await.unwrap();
    });
    format!("http://{addr}")
}

fn fast_client(url: String, key: &str) -> CoordinatorClient {
    CoordinatorClient::new(url, Wallet::from_hex(key).unwrap()).with_backoff(Backoff {
        initial: std::time::Duration::from_millis(5),
        max: std::time::Duration::from_millis(10),
    })
}

#[tokio::test]
async fn the_real_worker_client_completes_the_whole_loop_against_the_real_server() {
    let c = coordinator(
        "realclient",
        vec![JobSpec::new(
            "h01-64m-nat-seed1",
            Capability::H01,
            serde_json::json!({ "rung": "64M", "arm": "nat", "seed": 1 }),
        )],
    );
    let url = serve(c.clone()).await;
    let client = fast_client(url, KEY_A);

    // register — the capability is derived from the probe, not claimed
    let reg = client
        .register(&probe("candle-cuda", "f32", 71_098.0, true))
        .await
        .expect("register");
    assert_eq!(reg.capability, Capability::H01);
    // The server's idea of who we are matches ours, with no id ever transmitted.
    assert_eq!(reg.worker, format!("{:?}", client.worker_id()));

    // lease
    let job = client.lease().await.expect("lease").expect("a job");
    assert_eq!(job.id.0, "h01-64m-nat-seed1");
    assert_eq!(job.payload["rung"], "64M");

    // submit
    client
        .submit(&job.id, r#"{"final_loss":2.31}"#)
        .await
        .expect("submit");

    let counts = c.snapshot().counts();
    assert_eq!((counts.done, counts.pending, counts.workers), (1, 0, 1));
}

/// The capability gate, end to end through the real client: a CPU machine is told
/// there is nothing for it rather than handed ablation work.
#[tokio::test]
async fn a_cpu_client_is_given_no_work_by_the_real_server() {
    let c = coordinator(
        "realcpu",
        vec![JobSpec::new(
            "ladder",
            Capability::H01,
            serde_json::json!({}),
        )],
    );
    let client = fast_client(serve(c).await, KEY_A);

    let reg = client
        .register(&probe("candle-cpu", "f32", 7_786.0, true))
        .await
        .expect("register");
    assert_eq!(reg.capability, Capability::Probe);
    assert!(client
        .lease()
        .await
        .expect("lease is not an error")
        .is_none());
}

/// The poll loop is what actually runs on a member's machine: it must pick work
/// up, run it, and return a result the server accepts, unattended.
#[tokio::test]
async fn the_poll_loop_drains_the_queue_unattended() {
    let c = coordinator(
        "drain",
        vec![
            JobSpec::new("a", Capability::Probe, serde_json::json!({})),
            JobSpec::new("b", Capability::Probe, serde_json::json!({})),
        ],
    );
    let client = fast_client(serve(c.clone()).await, KEY_A);
    client
        .register(&probe("candle-cpu", "f32", 7_137.0, true))
        .await
        .unwrap();

    let done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let d2 = done.clone();
    client
        .poll_loop(
            move |job| {
                let d = d2.clone();
                async move {
                    d.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(format!(r#"{{"job":"{}"}}"#, job.id))
                }
            },
            {
                let d3 = done.clone();
                move || d3.load(std::sync::atomic::Ordering::SeqCst) < 2
            },
        )
        .await
        .unwrap();

    let counts = c.snapshot().counts();
    assert_eq!(counts.done, 2, "both jobs completed unattended");
    assert_eq!(counts.pending, 0);
}
