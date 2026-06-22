//! FWA-BV-CP-01 — redirect-follow on coordinator outbound (RED→GREEN).
//!
//! The coordinator's member-dispatch client (built inside
//! `handle_event`) and the JSON-RPC write client
//! (`HttpChainAdapter::new`) used reqwest's DEFAULT redirect policy
//! (follow up to 10). A pool member (or a MITM) answering the
//! `/pool-infer` POST with `307 Location: http://<off-gate>/sink`
//! would make reqwest re-POST the buyer prompt — in cleartext — to a
//! host that never passed the construction-time outbound TLS gate
//! (`outbound::validate_outbound_url`). The gate only validates the
//! initial URL string, so a 3xx downgrade slips past it.
//!
//! The fix sets `redirect(Policy::none())` on both outbound builders:
//! a 3xx comes back as a response (non-2xx → `ProviderFailed`) instead
//! of being silently followed, so the prompt body is never delivered
//! to the redirect target.
//!
//! This test stands up two local servers:
//!   - a *redirector* on `/pool-infer` that 307s to the sink, and
//!   - a *sink* that records any body it receives (the "off-gate host").
//!
//! It then drives the full `handle_event` path and asserts the sink
//! received NOTHING and the job was FAILED (not completed/paid).
//!
//! RED (default redirect policy): the client follows the 307, re-POSTs
//! the prompt to the sink → sink records 1 body, job completes → FAIL.
//! GREEN (Policy::none()): the 307 surfaces as a non-2xx →
//! `ProviderFailed` → job failed, sink records 0 bodies → PASS.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::post;
use ethereum_types::{H160, H256, U256};
use tokio::net::TcpListener;

use citrate_pool_coordinator::chain::{
    ChainAdapter, ComputeRequestedEvent, PoolMemberInfo, RecordDispatchOutcome,
};
use citrate_pool_coordinator::{handle_event, CoordinatorConfig, CoordinatorError};

// ── Sink server: records every body POSTed to it (the off-gate host) ──

async fn spawn_sink() -> (SocketAddr, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let app = axum::Router::new()
        .route(
            "/sink",
            post(|State(h): State<Arc<AtomicUsize>>, _body: String| async move {
                // Any body delivered here is a prompt leak past the gate.
                h.fetch_add(1, Ordering::SeqCst);
                axum::Json(serde_json::json!({
                    "output": "LEAKED-TO-SINK",
                    "input_tokens": 1,
                    "output_tokens": 1,
                }))
            }),
        )
        .with_state(hits.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind sink");
    let addr = listener.local_addr().expect("sink addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("sink serve");
    });
    (addr, hits)
}

// ── Redirector: answers /pool-infer with 307 → http://<sink>/sink ──

async fn spawn_redirector(sink: SocketAddr) -> SocketAddr {
    let location = format!("http://{}/sink", sink);
    let app = axum::Router::new().route(
        "/pool-infer",
        post(move |_body: String| {
            let location = location.clone();
            async move {
                (
                    axum::http::StatusCode::TEMPORARY_REDIRECT,
                    [(axum::http::header::LOCATION, location)],
                )
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind redirector");
    let addr = listener.local_addr().expect("redirector addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("redirector serve");
    });
    addr
}

// ── Minimal mock chain (records complete/fail) ───────────────────────

#[derive(Default, Clone)]
struct Calls {
    complete_job: Vec<u64>,
    fail_job: Vec<u64>,
}

struct MockChain {
    self_address: H160,
    members: Vec<PoolMemberInfo>,
    calls: Arc<Mutex<Calls>>,
}

#[async_trait]
impl ChainAdapter for MockChain {
    fn self_address(&self) -> H160 {
        self.self_address
    }
    async fn coordinator_for(&self, _pool: u64, _epoch: u64) -> Result<H160, CoordinatorError> {
        Ok(self.self_address) // we ARE the coordinator
    }
    async fn pool_members(&self, _pool: u64) -> Result<Vec<PoolMemberInfo>, CoordinatorError> {
        Ok(self.members.clone())
    }
    async fn record_dispatch(
        &self,
        _job: u64,
        _member: H160,
    ) -> Result<RecordDispatchOutcome, CoordinatorError> {
        Ok(RecordDispatchOutcome::Confirmed {
            tx_hash: H256::zero(),
            block_number: 1,
        })
    }
    async fn complete_job(&self, job: u64) -> Result<H256, CoordinatorError> {
        self.calls.lock().expect("mutex").complete_job.push(job);
        Ok(H256::zero())
    }
    async fn fail_job(&self, job: u64) -> Result<H256, CoordinatorError> {
        self.calls.lock().expect("mutex").fail_job.push(job);
        Ok(H256::zero())
    }
}

/// FWA-BV-CP-01: the dispatch client must NOT follow a 307 redirect
/// that downgrades to an off-gate sink, so the buyer prompt is never
/// re-POSTed in cleartext to a host the outbound gate never checked.
#[tokio::test]
async fn dispatch_does_not_follow_redirect_to_off_gate_sink() {
    let (sink_addr, sink_hits) = spawn_sink().await;
    let redirector = spawn_redirector(sink_addr).await;

    let member_addr = H160::from([0xb1; 20]);
    let calls = Arc::new(Mutex::new(Calls::default()));
    let chain = MockChain {
        self_address: H160::from([0xaa; 20]),
        members: vec![PoolMemberInfo {
            address: member_addr,
            gpu_count: 1,
            active: true,
        }],
        calls: calls.clone(),
    };

    let cfg = CoordinatorConfig {
        chain_id: 40204,
        rpc_url: "http://unused".to_string(),
        wallet_address: H160::from([0xaa; 20]),
        member_endpoints: [(member_addr, format!("http://{}/pool-infer", redirector))]
            .into_iter()
            .collect(),
        provider_timeout_secs: 30,
    };

    let event = ComputeRequestedEvent {
        pool_id: 1,
        job_id: 1234,
        requester: H160::from([0xc1; 20]),
        payment_grains: U256::from(1u64),
        prompt: "SECRET-BUYER-PROMPT".to_string(),
        max_tokens: 16,
        tx_hash: H256::zero(),
        log_index: 0,
        block_number: 1,
    };

    let outcome = handle_event(&chain, &cfg, &event).await;

    // The 307 must surface as a provider failure, not a silent follow.
    assert!(
        matches!(outcome, Err(CoordinatorError::ProviderFailed(_))),
        "a 3xx redirect must NOT be followed; expected ProviderFailed, got {:?}",
        outcome
    );

    // The prompt must NEVER have reached the off-gate sink.
    assert_eq!(
        sink_hits.load(Ordering::SeqCst),
        0,
        "buyer prompt was re-POSTed to the off-gate redirect sink (BV-CP-01 still vulnerable)"
    );

    // And the job must be FAILED, never completed (paid) off a 3xx.
    let calls = calls.lock().expect("mutex").clone();
    assert!(
        calls.complete_job.is_empty(),
        "job must not be completed/paid off a redirect response"
    );
    assert_eq!(calls.fail_job, vec![1234], "job must be failed on the 3xx");
}
