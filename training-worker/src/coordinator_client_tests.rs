// The client's half of the protocol, against a stub server.
//
// The full loop against the REAL coordinator lives in the coordinator crate,
// which is the only one allowed to depend on both. What is tested here is the
// behaviour the coordinator cannot exercise: that a 204 is not an error, that a
// coordinator being down does not kill the worker, and that the bytes put on the
// wire are the ones the shared digests define.

use super::*;
use crate::coordinator_protocol::Capability;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// Anvil account #0 — a well-known throwaway. NOT a real key.
const KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

fn wallet() -> Wallet {
    Wallet::from_hex(KEY).expect("load test key")
}

/// A minimal HTTP/1.1 server that replays canned responses and records requests.
/// Hand-rolled rather than pulled in as a dependency: the worker crate has no
/// server framework and does not need one for four routes' worth of assertions.
struct StubServer {
    addr: std::net::SocketAddr,
    seen: Arc<parking_lot::Mutex<Vec<(String, String)>>>,
    hits: Arc<AtomicUsize>,
}

impl StubServer {
    /// `replies` is consumed in order; the last one repeats once exhausted.
    async fn start(replies: Vec<(u16, &'static str)>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        let (s2, h2) = (seen.clone(), hits.clone());

        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let (s3, h3, replies) = (s2.clone(), h2.clone(), replies.clone());
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let Ok(n) = sock.read(&mut buf).await else { return };
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let path = req
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("")
                        .to_string();
                    let body = req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                    s3.lock().push((path, body));

                    let i = h3.fetch_add(1, Ordering::SeqCst);
                    let (code, payload) = replies[i.min(replies.len() - 1)];
                    let reason = if code == 204 { "No Content" } else { "OK" };
                    let res = format!(
                        "HTTP/1.1 {code} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    let _ = sock.write_all(res.as_bytes()).await;
                    let _ = sock.flush().await;
                });
            }
        });
        Self { addr, seen, hits }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn requests(&self) -> Vec<(String, String)> {
        self.seen.lock().clone()
    }

    fn hit_count(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

fn client(url: String) -> CoordinatorClient {
    CoordinatorClient::new(url, wallet()).with_backoff(Backoff {
        initial: Duration::from_millis(5),
        max: Duration::from_millis(20),
    })
}

// ── Paths and payloads ─────────────────────────────────────────────────

#[tokio::test]
async fn register_posts_the_probe_verbatim_with_a_recoverable_signature() {
    let probe = r#"{"schema":"nat.divergence-probe/1","backend":"candle-cuda"}"#;
    let srv = StubServer::start(vec![(200, r#"{"worker":"0xabc","capability":"h01"}"#)]).await;
    let r = client(srv.url()).register(probe).await.unwrap();
    assert_eq!(r.capability, Capability::H01);

    let (path, body) = srv.requests().into_iter().next().unwrap();
    assert_eq!(path, "/v1/register");
    let sent: serde_json::Value = serde_json::from_str(&body).unwrap();
    // Verbatim: a re-serialization would not match the signed digest.
    assert_eq!(sent["probe_json"], probe);
    // And the signature recovers to us, which is what the coordinator will check.
    let sig = hex::decode(sent["signature"].as_str().unwrap().trim_start_matches("0x")).unwrap();
    assert_eq!(
        Wallet::recover_address(&attestation_digest(probe), &sig).unwrap(),
        wallet().address()
    );
}

#[tokio::test]
async fn submit_signs_the_shared_digest_so_the_coordinator_can_recover_us() {
    let srv = StubServer::start(vec![(202, "")]).await;
    let job = JobId("h01-64m-seed2".into());
    client(srv.url()).submit(&job, r#"{"loss":2.3}"#).await.unwrap();

    let (path, body) = srv.requests().into_iter().next().unwrap();
    assert_eq!(path, "/v1/submit");
    let sent: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(sent["job"], "h01-64m-seed2");
    let sig = hex::decode(sent["signature"].as_str().unwrap().trim_start_matches("0x")).unwrap();
    assert_eq!(
        Wallet::recover_address(&submission_digest(&job, r#"{"loss":2.3}"#), &sig).unwrap(),
        wallet().address()
    );
}

#[tokio::test]
async fn a_trailing_slash_on_the_base_url_does_not_produce_a_double_slash() {
    let srv = StubServer::start(vec![(204, "")]).await;
    let c = CoordinatorClient::new(format!("{}/", srv.url()), wallet());
    c.lease().await.unwrap();
    assert_eq!(srv.requests()[0].0, "/v1/lease");
}

// ── 204 is not an error ────────────────────────────────────────────────

/// The steady state of a fleet with more machines than queued work. Treating it
/// as an error would fill logs and trip alerting on a healthy system.
#[tokio::test]
async fn no_work_is_ok_none_not_an_error() {
    let srv = StubServer::start(vec![(204, "")]).await;
    assert!(client(srv.url()).lease().await.unwrap().is_none());
}

#[tokio::test]
async fn a_job_is_returned_when_there_is_one() {
    let body = r#"{"id":"j1","requires":"probe","payload":{"k":1},"lease_secs":60,"max_attempts":3}"#;
    let srv = StubServer::start(vec![(200, body)]).await;
    let job = client(srv.url()).lease().await.unwrap().unwrap();
    assert_eq!(job.id.0, "j1");
    assert_eq!(job.requires, Capability::Probe);
    assert_eq!(job.lease_secs, 60);
}

#[tokio::test]
async fn a_rejection_carries_the_status_so_it_can_be_diagnosed() {
    let srv = StubServer::start(vec![(403, r#"{"error":"worker is not registered"}"#)]).await;
    match client(srv.url()).lease().await {
        Err(ClientError::Rejected { status, body }) => {
            assert_eq!(status, 403);
            assert!(body.contains("not registered"));
        }
        other => panic!("expected a rejection, got {other:?}"),
    }
}

// ── Surviving a fleet's normal day ─────────────────────────────────────

/// A worker that dies when the coordinator restarts is a worker a member has to
/// babysit, and they will not.
#[tokio::test]
async fn an_unreachable_coordinator_is_a_transport_error_not_a_panic() {
    // Nothing is listening on this port.
    let c = client("http://127.0.0.1:1".into());
    assert!(matches!(c.lease().await, Err(ClientError::Transport(_))));
}

#[tokio::test]
async fn the_poll_loop_keeps_going_when_the_coordinator_is_down() {
    let c = client("http://127.0.0.1:1".into());
    let rounds = Arc::new(AtomicUsize::new(0));
    let r2 = rounds.clone();
    c.poll_loop(
        |_job| async { Ok(String::new()) },
        move || r2.fetch_add(1, Ordering::SeqCst) < 3,
    )
    .await
    .unwrap();
    // It kept polling rather than returning on the first failure.
    assert_eq!(rounds.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn the_poll_loop_runs_a_job_and_submits_the_result() {
    let job = r#"{"id":"j1","requires":"probe","payload":{},"lease_secs":60,"max_attempts":3}"#;
    let srv = StubServer::start(vec![(200, job), (202, "")]).await;
    let c = client(srv.url());
    let done = Arc::new(AtomicUsize::new(0));
    let d2 = done.clone();

    c.poll_loop(
        move |j| {
            let d = d2.clone();
            async move {
                assert_eq!(j.id.0, "j1");
                d.fetch_add(1, Ordering::SeqCst);
                Ok(r#"{"loss":1.0}"#.to_string())
            }
        },
        {
            let d3 = done.clone();
            move || d3.load(Ordering::SeqCst) == 0
        },
    )
    .await
    .unwrap();

    assert_eq!(done.load(Ordering::SeqCst), 1);
    let paths: Vec<String> = srv.requests().into_iter().map(|(p, _)| p).collect();
    assert_eq!(paths, vec!["/v1/lease", "/v1/submit"]);
}

/// A machine that cannot do a job must not submit garbage for it. Letting the
/// lease expire hands the work to someone else, which is the coordinator's
/// `failed_by` path.
#[tokio::test]
async fn a_failed_job_is_not_submitted() {
    let job = r#"{"id":"j1","requires":"probe","payload":{},"lease_secs":60,"max_attempts":3}"#;
    let srv = StubServer::start(vec![(200, job), (204, "")]).await;
    let c = client(srv.url());
    let tried = Arc::new(AtomicUsize::new(0));
    let t2 = tried.clone();

    c.poll_loop(
        move |_j| {
            let t = t2.clone();
            async move {
                t.fetch_add(1, Ordering::SeqCst);
                Err(anyhow::anyhow!("out of VRAM"))
            }
        },
        {
            let t3 = tried.clone();
            move || t3.load(Ordering::SeqCst) == 0
        },
    )
    .await
    .unwrap();

    assert_eq!(tried.load(Ordering::SeqCst), 1);
    let paths: Vec<String> = srv.requests().into_iter().map(|(p, _)| p).collect();
    assert!(!paths.contains(&"/v1/submit".to_string()), "must not submit a failed job");
}

// ── Backoff ────────────────────────────────────────────────────────────

#[tokio::test]
async fn backoff_doubles_and_then_holds_at_the_cap() {
    let b = Backoff {
        initial: Duration::from_secs(5),
        max: Duration::from_secs(20),
    };
    let mut d = b.initial;
    let mut seen = vec![d];
    for _ in 0..4 {
        d = b.next(d);
        seen.push(d);
    }
    assert_eq!(
        seen,
        vec![
            Duration::from_secs(5),
            Duration::from_secs(10),
            Duration::from_secs(20),
            Duration::from_secs(20),
            Duration::from_secs(20),
        ]
    );
}

/// An idle fleet must not hammer the coordinator: consecutive empty polls have to
/// slow down rather than spin.
#[tokio::test]
async fn an_idle_worker_backs_off_instead_of_spinning() {
    let srv = StubServer::start(vec![(204, "")]).await;
    let c = client(srv.url());
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();

    let started = std::time::Instant::now();
    c.poll_loop(
        |_j| async { Ok(String::new()) },
        move || n2.fetch_add(1, Ordering::SeqCst) < 4,
    )
    .await
    .unwrap();

    // 5ms + 10ms + 20ms + 20ms of sleeping, so it cannot have spun through.
    assert!(started.elapsed() >= Duration::from_millis(50), "did not back off");
    assert!(srv.hit_count() >= 4);
}
