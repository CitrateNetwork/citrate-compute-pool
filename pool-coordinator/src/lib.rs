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
    // 1. Are we the elected coordinator for the current epoch?
    //    Slice 1 derives the epoch from the event's job_id-adjacent
    //    block; slice 2 reads `block.number` from the chain. For
    //    this slice we use a fixed epoch = 0 since the mock chain
    //    answers the same regardless.
    let epoch = 0; // slice 2 reads from the chain
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
    let chosen = select_member(event.job_id, &active)
        .ok_or(CoordinatorError::NoMembers(event.pool_id))?;

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
    let http = reqwest::Client::new();
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
