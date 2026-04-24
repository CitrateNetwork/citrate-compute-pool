//! WP-05.2 smoke test — pool-coordinator daemon, written FIRST.
//!
//! Exercises the daemon's core decision loop without spinning up a
//! real Citrate node. The chain interactions go through the
//! `ChainAdapter` trait, mocked here. Provider HTTPS goes through a
//! real axum stub on a random port.
//!
//! Slice 1 scope:
//!   - Coordinator role check via mocked `coordinator_for(pool, epoch)`
//!   - Stateless round-robin selection: `hash(job_id) % active_member_count`
//!   - HTTPS POST to `/pool-infer` against a stub provider
//!   - Mocked recordDispatch + completeJob tx submission
//!   - Failure paths: not the coordinator (skip), provider 5xx
//!     (record then fail rather than complete)
//!
//! Spec: .agentile/formal/specs/compute/InferencePoolLifecycle.tla
//! Behaviors: citrate_v0.01.1/specs/gherkin/inference_pool.feature

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::extract::Json as JsonExtractor;
use axum::routing::post;
use axum::Json as JsonResp;
use ethereum_types::{H160, U256};
use serde_json::Value;
use tokio::net::TcpListener;

use citrate_pool_coordinator::chain::{
    ChainAdapter, ComputeRequestedEvent, PoolMemberInfo, RecordDispatchOutcome,
};
use citrate_pool_coordinator::dispatcher::{select_member, MemberId};
use citrate_pool_coordinator::{handle_event, CoordinatorConfig, CoordinatorError};

// ── Stub provider — echoes prompt back as a canned response ────

async fn spawn_stub_provider() -> SocketAddr {
    let app = axum::Router::new().route(
        "/pool-infer",
        post(|JsonExtractor(req): JsonExtractor<Value>| async move {
            let prompt = req
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            JsonResp(serde_json::json!({
                "output": format!("STUB-RESPONSE: {}", prompt),
                "input_tokens": prompt.split_whitespace().count(),
                "output_tokens": 4,
            }))
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind stub");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serve");
    });
    addr
}

async fn spawn_failing_provider() -> SocketAddr {
    let app = axum::Router::new().route(
        "/pool-infer",
        post(|| async {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                JsonResp(serde_json::json!({"error": "stub-fail"})),
            )
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind stub");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serve");
    });
    addr
}

// ── Mock chain adapter ──────────────────────────────────────────

#[derive(Default, Clone)]
struct ChainCalls {
    record_dispatch: Vec<(u64, H160)>,
    complete_job: Vec<u64>,
    fail_job: Vec<u64>,
}

struct MockChain {
    /// Address this daemon's wallet signs as.
    self_address: H160,
    /// What `coordinator_for(pool, epoch)` returns. None means "not the coordinator."
    coordinator: Option<H160>,
    /// Members of every queried pool (no per-pool indexing for slice 1).
    members: Vec<PoolMemberInfo>,
    /// Whether dispatch should succeed.
    dispatch_ok: bool,
    calls: Arc<Mutex<ChainCalls>>,
}

impl MockChain {
    fn new(self_address: H160) -> Self {
        Self {
            self_address,
            coordinator: None,
            members: vec![],
            dispatch_ok: true,
            calls: Arc::new(Mutex::new(ChainCalls::default())),
        }
    }
}

#[async_trait]
impl ChainAdapter for MockChain {
    fn self_address(&self) -> H160 {
        self.self_address
    }

    async fn coordinator_for(
        &self,
        _pool_id: u64,
        _epoch: u64,
    ) -> Result<H160, CoordinatorError> {
        self.coordinator
            .ok_or_else(|| CoordinatorError::Chain("no coordinator".into()))
    }

    async fn pool_members(
        &self,
        _pool_id: u64,
    ) -> Result<Vec<PoolMemberInfo>, CoordinatorError> {
        Ok(self.members.clone())
    }

    async fn record_dispatch(
        &self,
        job_id: u64,
        member: H160,
    ) -> Result<RecordDispatchOutcome, CoordinatorError> {
        self.calls
            .lock()
            .expect("mutex")
            .record_dispatch
            .push((job_id, member));
        if self.dispatch_ok {
            Ok(RecordDispatchOutcome::Confirmed {
                tx_hash: ethereum_types::H256::zero(),
                block_number: 1,
            })
        } else {
            Err(CoordinatorError::Chain("recordDispatch reverted".into()))
        }
    }

    async fn complete_job(&self, job_id: u64) -> Result<ethereum_types::H256, CoordinatorError> {
        self.calls.lock().expect("mutex").complete_job.push(job_id);
        Ok(ethereum_types::H256::zero())
    }

    async fn fail_job(&self, job_id: u64) -> Result<ethereum_types::H256, CoordinatorError> {
        self.calls.lock().expect("mutex").fail_job.push(job_id);
        Ok(ethereum_types::H256::zero())
    }
}

fn config(provider_addrs: &[(H160, SocketAddr)]) -> CoordinatorConfig {
    let endpoints = provider_addrs
        .iter()
        .map(|(addr, sock)| (*addr, format!("http://{}/pool-infer", sock)))
        .collect();
    CoordinatorConfig {
        chain_id: 40204,
        rpc_url: "http://unused".to_string(),
        wallet_address: H160::from([0xaa; 20]),
        member_endpoints: endpoints,
        provider_timeout_secs: 30,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn happy_path_dispatches_records_completes() {
    let provider = spawn_stub_provider().await;
    let member_addr = H160::from([0xb1; 20]);

    let mut chain = MockChain::new(H160::from([0xaa; 20]));
    chain.coordinator = Some(H160::from([0xaa; 20])); // we ARE the coordinator
    chain.members = vec![PoolMemberInfo {
        address: member_addr,
        gpu_count: 1,
        active: true,
    }];
    let calls = chain.calls.clone();

    let cfg = config(&[(member_addr, provider)]);
    let event = ComputeRequestedEvent {
        pool_id: 1,
        job_id: 42,
        requester: H160::from([0xc1; 20]),
        payment_grains: U256::from(1_000_000_000_000_000_000u128),
        prompt: "ping".to_string(),
        max_tokens: 16,
    };

    let outcome = handle_event(&chain, &cfg, &event).await;
    assert!(outcome.is_ok(), "happy path: {:?}", outcome);

    let calls = calls.lock().expect("mutex").clone();
    assert_eq!(calls.record_dispatch, vec![(42, member_addr)]);
    assert_eq!(calls.complete_job, vec![42]);
    assert!(calls.fail_job.is_empty());
}

#[tokio::test]
async fn skips_event_when_not_coordinator() {
    let provider = spawn_stub_provider().await;
    let member_addr = H160::from([0xb1; 20]);

    let mut chain = MockChain::new(H160::from([0xaa; 20]));
    chain.coordinator = Some(H160::from([0xff; 20])); // someone ELSE
    chain.members = vec![PoolMemberInfo {
        address: member_addr,
        gpu_count: 1,
        active: true,
    }];
    let calls = chain.calls.clone();

    let cfg = config(&[(member_addr, provider)]);
    let event = ComputeRequestedEvent {
        pool_id: 1,
        job_id: 1,
        requester: H160::from([0xc1; 20]),
        payment_grains: U256::from(1u64),
        prompt: "x".to_string(),
        max_tokens: 1,
    };

    let outcome = handle_event(&chain, &cfg, &event).await;
    assert!(matches!(outcome, Err(CoordinatorError::NotCoordinator)));
    let calls = calls.lock().expect("mutex").clone();
    assert!(calls.record_dispatch.is_empty(), "no dispatch when not coord");
    assert!(calls.complete_job.is_empty());
}

#[tokio::test]
async fn provider_failure_marks_job_failed() {
    let provider = spawn_failing_provider().await;
    let member_addr = H160::from([0xb1; 20]);

    let mut chain = MockChain::new(H160::from([0xaa; 20]));
    chain.coordinator = Some(H160::from([0xaa; 20]));
    chain.members = vec![PoolMemberInfo {
        address: member_addr,
        gpu_count: 1,
        active: true,
    }];
    let calls = chain.calls.clone();

    let cfg = config(&[(member_addr, provider)]);
    let event = ComputeRequestedEvent {
        pool_id: 1,
        job_id: 7,
        requester: H160::from([0xc1; 20]),
        payment_grains: U256::from(1u64),
        prompt: "ping".to_string(),
        max_tokens: 1,
    };

    let outcome = handle_event(&chain, &cfg, &event).await;
    assert!(matches!(outcome, Err(CoordinatorError::ProviderFailed(_))));
    let calls = calls.lock().expect("mutex").clone();
    assert_eq!(calls.record_dispatch, vec![(7, member_addr)]);
    assert!(calls.complete_job.is_empty(), "no complete on provider fail");
    assert_eq!(calls.fail_job, vec![7]);
}

// ── Stateless round-robin ───────────────────────────────────────

#[tokio::test]
async fn select_member_is_deterministic_for_a_given_job_id() {
    let members = vec![
        MemberId(H160::from([0xa1; 20])),
        MemberId(H160::from([0xa2; 20])),
        MemberId(H160::from([0xa3; 20])),
    ];
    let a = select_member(42, &members).expect("non-empty");
    let b = select_member(42, &members).expect("non-empty");
    assert_eq!(a, b, "same job_id → same member");
}

#[tokio::test]
async fn select_member_distributes_uniformly_across_jobs() {
    let members = vec![
        MemberId(H160::from([0xa1; 20])),
        MemberId(H160::from([0xa2; 20])),
        MemberId(H160::from([0xa3; 20])),
    ];
    let mut hits = std::collections::HashMap::<H160, u32>::new();
    for job_id in 0..900u64 {
        let m = select_member(job_id, &members).expect("non-empty");
        *hits.entry(m.0).or_default() += 1;
    }
    // With 900 jobs across 3 members, expect ~300 each. Allow ±20%
    // for hash dispersion.
    for count in hits.values() {
        assert!(*count > 240 && *count < 360, "skewed distribution: {}", count);
    }
    assert_eq!(hits.len(), 3, "all members got at least one job");
}

#[tokio::test]
async fn select_member_returns_none_for_empty_pool() {
    let members: Vec<MemberId> = vec![];
    assert!(select_member(0, &members).is_none());
}

// ── Helpers used by impl tests below ────────────────────────────

#[allow(dead_code)]
fn _silence_unused_imports() {
    let _: HashSet<u64> = HashSet::new();
}
