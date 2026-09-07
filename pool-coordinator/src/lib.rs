//! `citrate-pool-coordinator` — off-chain coordinator daemon.
//!
//! Every member of a `ComputePool` (CM-05) runs a copy of this
//! binary. It subscribes to `ComputeRequested` events and, when this
//! daemon's wallet IS the elected coordinator for the current epoch,
//! dispatches the job to a pool member via HTTPS, records the
//! dispatch on-chain, and submits the completion (or failure) tx.
//!
//! # Architecture
//!
//! ```text
//!   ComputeRequested event (chain)
//!         │
//!         ▼
//!   ┌─────────────────────────┐
//!   │ handle_event(...)       │
//!   │                         │
//!   │ 1. Am I the coordinator?│  → coordinator_for(pool, epoch)
//!   │    No → NotCoordinator  │
//!   │                         │
//!   │ 2. Pick a member        │  → select_member(job_id, members)
//!   │                         │
//!   │ 3. recordDispatch tx    │  → ChainAdapter::record_dispatch
//!   │                         │
//!   │ 4. POST /pool-infer     │  → provider::dispatch_to_member
//!   │    Failure → fail_job   │
//!   │                         │
//!   │ 5. completeJob tx       │  → ChainAdapter::complete_job
//!   └─────────────────────────┘
//! ```
//!
//! # Slice 1 scope (this commit)
//!
//! - Pure decision-loop function (`handle_event`) wrapped in a
//!   trait-driven test harness. No live RPC client.
//! - `ChainAdapter` trait is the seam — slice 2 supplies an
//!   `HttpChainAdapter` implementation against a real Citrate node.
//! - Stateless round-robin via `dispatcher::select_member`.
//!
//! # Slice 2 (next commit)
//!
//! - `HttpChainAdapter` against `eth_subscribe newHeads` +
//!   `eth_call` for `coordinatorFor` reads.
//! - secp256k1 wallet for signing tx submissions.
//! - `main.rs` event-loop: `loop { recv_event(); spawn(handle_event); }`
//! - End-to-end devnet test with three coordinators racing for the
//!   coordinator role across an epoch boundary.

#![forbid(unsafe_code)]
// `missing_docs` would force a comment on every wire-type field. The
// struct-level docstrings + the field names carry enough meaning for
// slice 1's API. Revisit when the public surface is finalised in
// slice 2 (HttpChainAdapter + main loop).

use std::time::Duration;

use ethereum_types::H160;

pub mod chain;
pub mod config;
pub mod dispatcher;
pub mod error;
pub mod http_chain;
pub mod metrics;
pub mod outbound;
pub mod provider;
pub mod wallet;
pub mod ws_chain;

pub use chain::{ChainAdapter, ComputeRequestedEvent, EPOCH_LENGTH};
pub use config::CoordinatorConfig;
pub use error::CoordinatorError;
pub use http_chain::HttpChainAdapter;
pub use wallet::{tx_hash_of_signed, Eip1559Tx, Wallet, WalletError};
pub use ws_chain::{EventStream, WsChainSubscriber};

use crate::dispatcher::{select_member, MemberId};
use crate::provider::{dispatch_to_member, PoolInferRequest};

/// Default model name embedded in the provider request when the
/// chain event doesn't carry one. Matches the gateway's default
/// (CM-03 WP-03.2). Will be replaced by a per-job model name once
/// the on-chain `PoolJobSpec.modelHash` resolution lands in slice 2.
const DEFAULT_MODEL: &str = "llama-3.1-8b";

/// Process one `ComputeRequested` event end-to-end.
///
/// Returns:
/// - `Ok(())` — we were the coordinator and the dispatch + completion
///   round-tripped successfully.
/// - `Err(NotCoordinator)` — we weren't the elected coordinator;
///   nothing was attempted on-chain.
/// - `Err(ProviderFailed(...))` — we recorded the dispatch but the
///   provider failed; we then submitted `failJob` so the buyer is
///   refunded. The error is surfaced for the caller to log/metric.
/// - `Err(...)` — chain RPC or config error before we could even
///   pick a member.
pub async fn handle_event<C: ChainAdapter + ?Sized>(
    chain: &C,
    cfg: &CoordinatorConfig,
    event: &ComputeRequestedEvent,
) -> Result<(), CoordinatorError> {
    // 0. CP-B-008: admission gate — refuse abusive events BEFORE any
    //    on-chain `recordDispatch`. The dispatch decision otherwise
    //    never reads escrow, never caps the requester-supplied prompt,
    //    and bounds `max_tokens` only at `u32::MAX`, so a zero-payment
    //    job demanding billions of tokens over a multi-megabyte prompt
    //    would be recorded on-chain against a deterministically-selected
    //    honest member and then `failJob`'d against it. Rejecting here
    //    means the member is never attributed the work at all.
    if !cfg.min_payment_grains.is_zero() && event.payment_grains < cfg.min_payment_grains {
        return Err(CoordinatorError::RejectedEvent(format!(
            "job {} payment_grains {} below floor {}",
            event.job_id, event.payment_grains, cfg.min_payment_grains
        )));
    }
    if event.prompt.len() > cfg.max_prompt_bytes {
        return Err(CoordinatorError::RejectedEvent(format!(
            "job {} prompt {} bytes exceeds cap {}",
            event.job_id,
            event.prompt.len(),
            cfg.max_prompt_bytes
        )));
    }
    if event.max_tokens > cfg.max_tokens_cap {
        return Err(CoordinatorError::RejectedEvent(format!(
            "job {} max_tokens {} exceeds cap {}",
            event.job_id, event.max_tokens, cfg.max_tokens_cap
        )));
    }

    // 1. Are we the elected coordinator for the current epoch?
    //    SECREM-02 6.3 (CITRATE_COMPUTE_POOL-2026-05-31-008): the
    //    epoch is derived from the event's block number, mirroring
    //    `block.number / EPOCH_LENGTH` in ComputePool.sol — the
    //    hardcoded epoch-0 election made every daemon query a stale
    //    election and race as duplicate coordinators after epoch 0.
    let epoch = crate::chain::epoch_of(event.block_number);
    let elected = chain.coordinator_for(event.pool_id, epoch).await?;
    if elected != chain.self_address() {
        tracing::debug!(
            event_pool = event.pool_id,
            event_job = event.job_id,
            elected = ?elected,
            self = ?chain.self_address(),
            "not the coordinator; skipping"
        );
        return Err(CoordinatorError::NotCoordinator);
    }

    // 2. Pick a member via stateless round-robin.
    let members = chain.pool_members(event.pool_id).await?;
    if members.is_empty() {
        return Err(CoordinatorError::NoMembers(event.pool_id));
    }
    let active: Vec<MemberId> = members
        .iter()
        .filter(|m| m.active && m.gpu_count > 0)
        .map(|m| MemberId(m.address))
        .collect();
    let chosen =
        select_member(event.job_id, &active).ok_or(CoordinatorError::NoMembers(event.pool_id))?;

    // 3. Resolve the chosen member to its HTTPS endpoint via config.
    let endpoint = cfg
        .member_endpoints
        .get(&chosen.0)
        .ok_or(CoordinatorError::UnknownMemberEndpoint(chosen.0))?
        .clone();

    // 4. Record the dispatch on-chain BEFORE making the HTTP call.
    //    The contract requires the recordDispatch caller be the
    //    elected coordinator AND the job be Pending — both true here.
    //    Recording first means: even if the HTTP call hangs and we
    //    crash, on restart the chain knows we dispatched and the
    //    coordination-timeout clock has started. Without this order
    //    a crash mid-call would leave the job Pending forever from
    //    the chain's perspective.
    chain.record_dispatch(event.job_id, chosen.0).await?;

    // 5. Dispatch the prompt to the chosen member.
    //    FWA-BV-CP-01: disable redirect-follow. The default policy
    //    follows up to 10 redirects and would re-POST the buyer prompt
    //    on a 307/308 to a `Location:` host that bypasses the
    //    construction-time outbound gate (e.g. https->http downgrade to
    //    a cleartext off-gate sink). `Policy::none()` returns the 3xx as
    //    a response instead, so the prompt is never delivered to it.
    // CP-B-012: do NOT fall back to `Client::default()` on a builder
    // error — the default follows up to 10 redirects, which is exactly
    // the FWA-BV-CP-01 vulnerability this `Policy::none()` exists to
    // prevent (fail-open). A coordinator that cannot build a
    // redirect-safe client must not dispatch the buyer prompt at all.
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| {
            CoordinatorError::Internal(format!("could not build redirect-safe HTTP client: {e}"))
        })?;
    let body = PoolInferRequest {
        model: DEFAULT_MODEL.to_string(),
        prompt: event.prompt.clone(),
        max_tokens: event.max_tokens,
        job_id: event.job_id,
    };
    let timeout = Duration::from_secs(cfg.provider_timeout_secs);
    let dispatch_result = dispatch_to_member(&http, &endpoint, &body, timeout).await;

    match dispatch_result {
        Ok(_resp) => {
            // 6a. Complete the job on-chain. Payment distributes to
            // pool members proportionally per `_distributePayment`.
            chain.complete_job(event.job_id).await?;
            tracing::info!(
                pool = event.pool_id,
                job = event.job_id,
                member = ?chosen.0,
                "job completed"
            );
            Ok(())
        }
        Err(provider_err) => {
            // 6b. Fail the job on-chain. Buyer is refunded. The
            // ChainAdapter's fail_job is best-effort: if it errors,
            // the original provider error is still what the operator
            // sees, and a follow-up reconciliation pass cleans up.
            tracing::warn!(
                pool = event.pool_id,
                job = event.job_id,
                member = ?chosen.0,
                error = %provider_err,
                "provider failed; marking job failed on-chain"
            );
            let _ = chain.fail_job(event.job_id).await;
            Err(provider_err)
        }
    }
}

/// Convenience: log this daemon's identity at startup.
pub fn log_identity(self_addr: H160, member_count: usize) {
    tracing::info!(
        wallet = ?self_addr,
        configured_members = member_count,
        "pool-coordinator starting"
    );
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::chain::{ComputeRequestedEvent, PoolMemberInfo, RecordDispatchOutcome};
    use async_trait::async_trait;
    use ethereum_types::{H160, H256, U256};

    /// A chain adapter that PANICS if `record_dispatch` is ever called.
    /// CP-B-008: the admission gate must refuse an abusive event before
    /// any on-chain attribution, so a rejected event must never reach
    /// `record_dispatch` (which is what marks an honest member the
    /// party that then gets `failJob`'d).
    struct PanicOnDispatch;

    #[async_trait]
    impl ChainAdapter for PanicOnDispatch {
        fn self_address(&self) -> H160 {
            H160::from([0xaa; 20])
        }
        async fn coordinator_for(&self, _p: u64, _e: u64) -> Result<H160, CoordinatorError> {
            Ok(H160::from([0xaa; 20]))
        }
        async fn pool_members(&self, _p: u64) -> Result<Vec<PoolMemberInfo>, CoordinatorError> {
            Ok(vec![PoolMemberInfo {
                address: H160::from([0xb1; 20]),
                gpu_count: 1,
                active: true,
            }])
        }
        async fn record_dispatch(
            &self,
            _job: u64,
            _m: H160,
        ) -> Result<RecordDispatchOutcome, CoordinatorError> {
            panic!("record_dispatch must NOT be reached for a rejected event (CP-B-008)");
        }
        async fn complete_job(&self, _j: u64) -> Result<H256, CoordinatorError> {
            Ok(H256::zero())
        }
        async fn fail_job(&self, _j: u64) -> Result<H256, CoordinatorError> {
            Ok(H256::zero())
        }
    }

    fn cfg() -> CoordinatorConfig {
        CoordinatorConfig {
            chain_id: 40204,
            rpc_url: "http://127.0.0.1:8545".to_string(),
            wallet_address: H160::from([0xaa; 20]),
            member_endpoints: std::collections::HashMap::new(),
            provider_timeout_secs: 30,
            min_payment_grains: U256::from(1u64),
            max_prompt_bytes: 128 * 1024,
            max_tokens_cap: 8192,
        }
    }

    fn event(payment: U256, prompt: String, max_tokens: u32) -> ComputeRequestedEvent {
        ComputeRequestedEvent {
            pool_id: 1,
            job_id: 7,
            requester: H160::from([0xc1; 20]),
            payment_grains: payment,
            prompt,
            max_tokens,
            tx_hash: H256::zero(),
            log_index: 0,
            block_number: 1,
        }
    }

    #[tokio::test]
    async fn zero_payment_event_is_rejected_before_dispatch() {
        let out = handle_event(
            &PanicOnDispatch,
            &cfg(),
            &event(U256::zero(), "hi".into(), 16),
        )
        .await;
        assert!(
            matches!(out, Err(CoordinatorError::RejectedEvent(_))),
            "got {out:?}"
        );
    }

    #[tokio::test]
    async fn oversized_prompt_is_rejected_before_dispatch() {
        let big = "x".repeat(128 * 1024 + 1);
        let out = handle_event(
            &PanicOnDispatch,
            &cfg(),
            &event(U256::from(1_000u64), big, 16),
        )
        .await;
        assert!(
            matches!(out, Err(CoordinatorError::RejectedEvent(_))),
            "got {out:?}"
        );
    }

    #[tokio::test]
    async fn over_cap_max_tokens_is_rejected_before_dispatch() {
        let out = handle_event(
            &PanicOnDispatch,
            &cfg(),
            &event(U256::from(1_000u64), "hi".into(), 4_294_967_295),
        )
        .await;
        assert!(
            matches!(out, Err(CoordinatorError::RejectedEvent(_))),
            "got {out:?}"
        );
    }
}
