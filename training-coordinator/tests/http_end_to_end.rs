//! The whole loop over the real router: register → lease → submit → status.
//!
//! The state machine is unit-tested in `state.rs`. This exists because the
//! interesting failures of a service are in the wiring — a route that recovers
//! the signer from the wrong bytes, or an error mapped to a status code that
//! makes a worker retry forever. None of those are visible to a unit test.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use citrate_training_coordinator::api::{heartbeat_digest, lease_digest, router, Coordinator};
use citrate_training_coordinator::job::{Capability, JobSpec};
use citrate_training_coordinator::state::Policy;
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
    // Unique per process and call, so two suite runs on one host (or two
    // tests sharing a name) never share a state file.
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::env::temp_dir().join(format!(
        "citrate-coord-http-{name}-{}-{n}",
        std::process::id()
    ));
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
    coordinator_trusting(name, jobs, &[])
}

/// A coordinator whose operator has vouched `keys` for H-01 work
/// (PBA-L3b-001: a self-reported H-01 probe alone no longer earns the ladder).
fn coordinator_trusting(name: &str, jobs: Vec<JobSpec>, keys: &[&str]) -> Arc<Coordinator> {
    let mut policy = Policy::default();
    for k in keys {
        policy
            .trusted_h01
            .insert(Wallet::from_hex(k).unwrap().address());
    }
    let c = Arc::new(Coordinator::open_with(Store::new(tmp(name)), policy).unwrap());
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

fn unix_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// A fresh, timestamp-bound lease request (CP-B-002). Each call signs a distinct
/// preimage, the way the real worker client does.
fn lease_body(key: &str) -> serde_json::Value {
    let ts = unix_nanos();
    serde_json::json!({
        "timestamp": ts,
        "signature": hex_sig(key, &lease_digest(ts)),
    })
}

#[tokio::test]
async fn a_worker_registers_leases_submits_and_the_status_reflects_it() {
    let c = coordinator_trusting(
        "happy",
        vec![JobSpec::new(
            "h01-64m-seed2",
            Capability::H01,
            serde_json::json!({"rung":"64M"}),
        )],
        &[KEY_A],
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
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
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
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
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
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
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
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
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
        .oneshot(post("/v1/lease", lease_body(KEY_B)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);
}

/// CP-B-002: a captured lease request must not be a forever-replayable bearer
/// credential. Queue two jobs, capture ONE signed lease body, and replay the
/// exact bytes: the coordinator leases at most one job and the replay is refused.
/// A stale timestamp is refused outright.
#[tokio::test]
async fn a_captured_lease_request_cannot_be_replayed() {
    let c = coordinator(
        "replay",
        vec![
            JobSpec::new("a", Capability::Probe, serde_json::json!({})),
            JobSpec::new("b", Capability::Probe, serde_json::json!({})),
        ],
    );
    let res = router(c.clone())
        .oneshot(post(
            "/v1/register",
            register_body(KEY_A, &probe("candle-cpu", "f32", 7_137.0, true)),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Capture exactly one signed lease body.
    let captured = lease_body(KEY_A);

    // First use: leases a job.
    let res = router(c.clone())
        .oneshot(post("/v1/lease", captured.clone()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Replay of the identical bytes: refused, and no second job is leased.
    let res = router(c.clone())
        .oneshot(post("/v1/lease", captured.clone()))
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::UNAUTHORIZED,
        "a replayed lease request must be refused"
    );

    let counts = c.snapshot().counts();
    assert_eq!(counts.leased, 1, "replay must not lease a second job");
    assert_eq!(counts.pending, 1);

    // A stale timestamp (far outside the freshness window) is refused outright.
    let stale_ts = 1_000_000_000u64; // ~1s after the unix epoch — ancient
    let stale = serde_json::json!({
        "timestamp": stale_ts,
        "signature": hex_sig(KEY_A, &lease_digest(stale_ts)),
    });
    let res = router(c.clone())
        .oneshot(post("/v1/lease", stale))
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::UNAUTHORIZED,
        "a stale lease request must be refused"
    );
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
    let c = coordinator_trusting(
        "realclient",
        vec![JobSpec::new(
            "h01-64m-nat-seed1",
            Capability::H01,
            serde_json::json!({ "rung": "64M", "arm": "nat", "seed": 1 }),
        )],
        &[KEY_A],
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

/// CP-B-009: `GET /v1/status` must be a true READ — it must never mutate
/// shared coordinator state without persisting it. Pre-fix, `status`
/// called `expire_leases` on the live state (rewriting JobStatus,
/// inserting into `failed_by`) while bypassing `Coordinator::mutate`, so
/// an unauthenticated GET silently diverged in-memory state from the
/// crash-atomic store; a restart in that window resurrected dead leases.
/// Post-fix, `status` expires on a snapshot, so after the GET the on-disk
/// state still equals the in-memory state.
#[tokio::test]
async fn status_get_does_not_diverge_memory_from_disk() {
    let path = tmp("status-readonly");
    let c = Arc::new(Coordinator::open(Store::new(path.clone())).unwrap());
    // A job whose lease expires immediately, so the very next status read
    // sees an expirable lease.
    c.add_job(
        JobSpec::new("probe-job", Capability::Probe, serde_json::json!({})).with_lease_secs(0),
    )
    .unwrap();

    // Register a probe-tier worker and lease the job.
    let res = router(c.clone())
        .oneshot(post(
            "/v1/register",
            register_body(KEY_A, &probe("candle-cpu", "f32", 7_786.0, true)),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = router(c.clone())
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Hit the read-only status route (the lease is now expired).
    let res = router(c.clone())
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // The store on disk must equal the in-memory state — status must not
    // have mutated shared state behind the store's back.
    let on_disk = Store::new(path).load().unwrap();
    let in_memory = c.snapshot();
    assert_eq!(
        serde_json::to_string(&on_disk).unwrap(),
        serde_json::to_string(&in_memory).unwrap(),
        "GET /v1/status diverged in-memory state from the persisted store (CP-B-009)"
    );
}

// ── PBA-L3b-001 over the real router ───────────────────────────────────

fn from_peer(mut req: Request<Body>, peer: &str) -> Request<Body> {
    let addr: std::net::SocketAddr = peer.parse().unwrap();
    req.extensions_mut()
        .insert(axum::extract::ConnectInfo(addr));
    req
}

/// A deterministic, valid throwaway key per index.
fn throwaway(i: u64) -> String {
    format!("0x{:064x}", 0x1000 + i)
}

/// Tripwire over HTTP (the audit PoC, at the real entry point). A fresh key per
/// cycle registers a forged-fast H-01 probe and polls first; the vouched honest
/// worker polls second. Before the fix the squatter was granted `h01` and took
/// the ladder every cycle; now it is granted `federated` and the honest worker
/// gets the ladder on the first cycle.
#[tokio::test]
async fn pba_l3b_001_fresh_h01_keys_cannot_squat_the_ladder_over_http() {
    let c = coordinator_trusting(
        "squat",
        vec![JobSpec::new("rung", Capability::H01, serde_json::json!({})).with_lease_secs(172_800)],
        &[KEY_A],
    );
    let fast = probe("candle-cuda", "f32", 71_098.0, true);
    for i in 0..3u64 {
        let k = throwaway(i);
        let res = router(c.clone())
            .oneshot(from_peer(
                post("/v1/register", register_body(&k, &fast)),
                "203.0.113.7:1",
            ))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(json_of(res).await["capability"], "probe");
        let res = router(c.clone())
            .oneshot(post("/v1/lease", lease_body(&k)))
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::NO_CONTENT,
            "squatter must not get the ladder"
        );
    }
    let res = router(c.clone())
        .oneshot(post("/v1/register", register_body(KEY_A, &fast)))
        .await
        .unwrap();
    assert_eq!(json_of(res).await["capability"], "h01");
    let res = router(c.clone())
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(json_of(res).await["id"], "rung");
}

/// Never-seen identities spend from a global budget; the burst is finite.
#[tokio::test]
async fn pba_l3b_001_new_identity_registration_is_rate_limited() {
    use citrate_training_coordinator::api::NEW_REGISTRATION_BURST;
    let c = coordinator("reg-budget", vec![]);
    let body = probe("candle-cpu", "f32", 7_786.0, true);
    let mut limited = None;
    for i in 0..(NEW_REGISTRATION_BURST + 5) {
        // A distinct public source each time, so only the global budget applies.
        let peer = format!("10.{}.{}.1:5000", i / 250, i % 250);
        let res = router(c.clone())
            .oneshot(from_peer(
                post("/v1/register", register_body(&throwaway(i), &body)),
                &peer,
            ))
            .await
            .unwrap();
        if res.status() == StatusCode::TOO_MANY_REQUESTS {
            limited = Some(i);
            break;
        }
        assert_eq!(res.status(), StatusCode::OK);
    }
    assert_eq!(limited, Some(NEW_REGISTRATION_BURST));
    // Refreshing an already-known identity is free even when over budget.
    let res = router(c.clone())
        .oneshot(from_peer(
            post("/v1/register", register_body(&throwaway(0), &body)),
            "10.0.0.1:1",
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

/// One client address can hold only so many identities; behind the loopback
/// proxy the address is the one Caddy put in `X-Forwarded-For`.
#[tokio::test]
async fn pba_l3b_001_one_client_address_is_capped_through_the_proxy() {
    let c = coordinator("per-source", vec![]);
    let cap = Policy::default().max_workers_per_source as u64;
    let body = probe("candle-cpu", "f32", 7_786.0, true);
    let reg = |i: u64, xff: &str| {
        let mut r = from_peer(
            post("/v1/register", register_body(&throwaway(i), &body)),
            "127.0.0.1:40000",
        );
        r.headers_mut()
            .insert("x-forwarded-for", xff.parse().unwrap());
        r
    };
    for i in 0..cap {
        let res = router(c.clone())
            .oneshot(reg(i, "198.51.100.9"))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "identity {i}");
    }
    let res = router(c.clone())
        .oneshot(reg(cap, "198.51.100.9"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    // A spoofed leftmost entry does not help: the proxy-written (rightmost) one counts.
    let res = router(c.clone())
        .oneshot(reg(cap, "1.2.3.4, 198.51.100.9"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    // A different client is unaffected.
    let res = router(c.clone())
        .oneshot(reg(cap, "198.51.100.10"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[test]
fn pba_l3b_001_forwarded_for_is_believed_only_from_a_loopback_peer() {
    use citrate_training_coordinator::api::source_of;
    let mut h = axum::http::HeaderMap::new();
    h.insert("x-forwarded-for", "9.9.9.9, 198.51.100.9".parse().unwrap());
    let lo: std::net::SocketAddr = "127.0.0.1:1".parse().unwrap();
    let public: std::net::SocketAddr = "203.0.113.5:1".parse().unwrap();
    assert_eq!(source_of(Some(lo), &h), "198.51.100.9");
    // A direct (non-proxied) client cannot choose its own source.
    assert_eq!(source_of(Some(public), &h), "203.0.113.5");
    // No peer information at all is never header-derived.
    assert_eq!(source_of(None, &h), "unknown");
    // Loopback with no header is the loopback host itself.
    assert_eq!(
        source_of(Some(lo), &axum::http::HeaderMap::new()),
        "127.0.0.1"
    );
    // IPv6 is folded to its /64: one host controls the whole prefix.
    let v6a: std::net::SocketAddr = "[2001:db8:1:2:aaaa::1]:1".parse().unwrap();
    let v6b: std::net::SocketAddr = "[2001:db8:1:2:bbbb::9]:1".parse().unwrap();
    assert_eq!(source_of(Some(v6a), &h), source_of(Some(v6b), &h));
    assert_eq!(source_of(Some(v6a), &h), "2001:db8:1:2::/64");
}

fn heartbeat_body(key: &str, job: &str) -> serde_json::Value {
    let ts = unix_nanos();
    serde_json::json!({
        "job": job,
        "timestamp": ts,
        "signature": hex_sig(key, &heartbeat_digest(&JobId(job.into()), ts)),
    })
}

/// The heartbeat route: the leaseholder extends its lease, anyone else is
/// refused, and a captured heartbeat cannot be replayed.
#[tokio::test]
async fn pba_l3b_001_heartbeat_route_renews_only_for_the_leaseholder() {
    let c = coordinator(
        "heartbeat",
        vec![JobSpec::new("j", Capability::Probe, serde_json::json!({})).with_lease_secs(172_800)],
    );
    let cpu = probe("candle-cpu", "f32", 7_786.0, true);
    for k in [KEY_A, KEY_B] {
        router(c.clone())
            .oneshot(post("/v1/register", register_body(k, &cpu)))
            .await
            .unwrap();
    }
    let res = router(c.clone())
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let hb = heartbeat_body(KEY_A, "j");
    let res = router(c.clone())
        .oneshot(post("/v1/heartbeat", hb.clone()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert!(json_of(res).await["expires_at"].as_u64().unwrap() > 0);

    let res = router(c.clone())
        .oneshot(post("/v1/heartbeat", hb))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "replayed heartbeat");

    let res = router(c.clone())
        .oneshot(post("/v1/heartbeat", heartbeat_body(KEY_B, "j")))
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::CONFLICT,
        "non-leaseholder heartbeat"
    );

    // A heartbeat signed for one job cannot be presented as another's.
    let ts = unix_nanos();
    let forged = serde_json::json!({
        "job": "j",
        "timestamp": ts,
        "signature": hex_sig(KEY_A, &heartbeat_digest(&JobId("other".into()), ts)),
    });
    let res = router(c.clone())
        .oneshot(post("/v1/heartbeat", forged))
        .await
        .unwrap();
    assert_ne!(
        res.status(),
        StatusCode::OK,
        "a heartbeat is bound to its job"
    );
}

/// A second lease while one is held is refused with 409, over the wire.
#[tokio::test]
async fn pba_l3b_001_second_concurrent_lease_is_refused_over_http() {
    let c = coordinator(
        "lease-cap",
        vec![
            JobSpec::new("a", Capability::Probe, serde_json::json!({})),
            JobSpec::new("b", Capability::Probe, serde_json::json!({})),
        ],
    );
    router(c.clone())
        .oneshot(post(
            "/v1/register",
            register_body(KEY_A, &probe("candle-cpu", "f32", 7_786.0, true)),
        ))
        .await
        .unwrap();
    let res = router(c.clone())
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = router(c.clone())
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CONFLICT);
    assert_eq!(c.snapshot().counts().leased, 1);
}

/// Drive the real client through one job of `job_ms` against a coordinator
/// with a `window_secs` renewal window, heartbeating every 2 s. `blocking`
/// models the job as synchronous CPU work (`std::thread::sleep`, no await
/// point), which is what the real candle executor is.
async fn run_one_heartbeated_job(
    name: &str,
    window_secs: u64,
    job_ms: u64,
    blocking: bool,
) -> usize {
    let policy = Policy {
        lease_window_secs: window_secs,
        ..Policy::default()
    };
    let c = Arc::new(Coordinator::open_with(Store::new(tmp(name)), policy).unwrap());
    c.add_job(
        JobSpec::new("slow", Capability::Probe, serde_json::json!({})).with_lease_secs(172_800),
    )
    .unwrap();
    let client = fast_client(serve(c.clone()).await, KEY_A)
        // Every heartbeat fsyncs the coordinator's state file; a 2 s cadence
        // keeps the test from saturating a busy disk with syncs.
        .with_heartbeat_every(std::time::Duration::from_secs(2));
    client
        .register(&probe("candle-cpu", "f32", 7_137.0, true))
        .await
        .unwrap();
    let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let r = ran.clone();
    let run = client.poll_loop(
        move |_job| {
            let r = r.clone();
            async move {
                if blocking {
                    std::thread::sleep(std::time::Duration::from_millis(job_ms));
                } else {
                    tokio::time::sleep(std::time::Duration::from_millis(job_ms)).await;
                }
                r.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok("{}".to_string())
            }
        },
        {
            let r = ran.clone();
            move || !r.load(std::sync::atomic::Ordering::SeqCst)
        },
    );
    // Never hang the suite: a lost lease makes the poll loop spin forever.
    tokio::time::timeout(std::time::Duration::from_secs(60), run)
        .await
        .expect("poll loop did not finish within 60 s")
        .unwrap();
    c.snapshot().counts().done
}

/// The real client heartbeats a running job, and that is what keeps the lease:
/// an 18 s job against a 15 s window is only accepted if heartbeats landed
/// (the window is generous because each heartbeat fsyncs the state file, which
/// can stall for seconds on a busy shared disk).
#[tokio::test]
async fn pba_l3b_001_the_real_client_heartbeats_while_a_job_runs() {
    assert_eq!(
        run_one_heartbeated_job("client-hb", 15, 18_000, false).await,
        1,
        "the heartbeated lease must still be live when the result is submitted"
    );
}

/// Verifier PoC `verify_blocking_job_never_heartbeats`, as the regression. The
/// real executor is synchronous and never yields; before the fix the heartbeat
/// shared its task and never fired, so every long job lapsed at the window.
/// Runs on the default current-thread runtime, the harshest case.
#[tokio::test]
async fn pba_l3b_001_a_blocking_job_is_still_heartbeated() {
    assert_eq!(
        run_one_heartbeated_job("client-hb-blocking", 15, 18_000, true).await,
        1,
        "a synchronous job lost its lease: heartbeats never fired"
    );
}

/// Same, on a multi-thread runtime (what `#[tokio::main]` builds for the worker).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pba_l3b_001_a_blocking_job_is_still_heartbeated_multi_thread() {
    assert_eq!(
        run_one_heartbeated_job("client-hb-blocking-mt", 15, 18_000, true).await,
        1,
        "a synchronous job lost its lease: heartbeats never fired"
    );
}

/// Verifier PoC `verify_one_host_drains_global_registration_budget`, as the
/// regression: 40 attempts from one full host must not spend the global
/// new-identity budget, so an honest newcomer elsewhere still gets in.
#[tokio::test]
async fn pba_l3b_001_a_full_source_cannot_drain_the_registration_budget() {
    let c = coordinator("budget-drain", vec![]);
    let body = |k: &str| register_body(k, &probe("candle-cpu", "f32", 7_000.0, true));
    let cap = Policy::default().max_workers_per_source as u64;
    for i in 0..40u64 {
        let res = router(c.clone())
            .oneshot(from_peer(
                post("/v1/register", body(&throwaway(i))),
                "203.0.113.7:1",
            ))
            .await
            .unwrap();
        let want = if i < cap {
            StatusCode::OK
        } else {
            StatusCode::TOO_MANY_REQUESTS
        };
        assert_eq!(res.status(), want, "attempt {i}");
    }
    let res = router(c.clone())
        .oneshot(from_peer(
            post("/v1/register", body(KEY_B)),
            "198.51.100.1:1",
        ))
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "honest onboarding denied by one host"
    );
}

/// A coordinator restarted with a tighter tier policy requeues leases the new
/// policy no longer allows, and persists that.
#[tokio::test]
async fn tier_restart_with_a_tighter_policy_requeues_disallowed_leases() {
    let path = tmp("tier-restart");
    let open = Policy {
        open_tier: Capability::Federated,
        ..Policy::default()
    };
    let first = Arc::new(Coordinator::open_with(Store::new(&path), open).unwrap());
    first
        .add_job(
            JobSpec::new("f", Capability::Federated, serde_json::json!({})).with_lease_secs(9999),
        )
        .unwrap();
    let fast = probe("candle-cuda", "f32", 71_098.0, true);
    router(first.clone())
        .oneshot(post("/v1/register", register_body(KEY_A, &fast)))
        .await
        .unwrap();
    let res = router(first.clone())
        .oneshot(post("/v1/lease", lease_body(KEY_A)))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    drop(first);

    let second = Arc::new(Coordinator::open_with(Store::new(&path), Policy::default()).unwrap());
    let c = second.snapshot().counts();
    assert_eq!(
        (c.leased, c.pending),
        (0, 1),
        "the federated lease is requeued at boot"
    );
    let on_disk = Store::new(&path).load().unwrap().counts();
    assert_eq!((on_disk.leased, on_disk.pending), (0, 1), "and persisted");
}
