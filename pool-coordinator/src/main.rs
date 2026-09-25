//! `citrate-pool-coordinator` — production binary entry point.
//!
//! Loads the coordinator's wallet, points an `HttpChainAdapter` at
//! a live Citrate JSON-RPC endpoint, and runs an event polling
//! loop against `ComputePool`'s `ComputeRequested` logs. For every
//! new event, spawns `handle_event` — if we're the elected
//! coordinator for the current epoch, dispatches to a pool member
//! via the existing provider protocol; otherwise silently skips.
//!
//! # Required env vars
//!
//! One of the two wallet sources (keystore is preferred for
//! production):
//!
//! - `CITRATE_POOL_KEYSTORE_PATH` + `CITRATE_POOL_KEYSTORE_PASSPHRASE`
//!   — Web3 SSv3 keystore file + unlock passphrase (production)
//! - `CITRATE_POOL_PRIVATE_KEY_HEX` — raw 64-char hex secp256k1 key
//!   (testnet only; rotate before production per pilot playbook §5.3)
//!
//! Plus:
//! - `CITRATE_POOL_WALLET_ADDRESS` — operator-declared address;
//!   MUST match derived address from the key
//! - `CITRATE_POOL_CONTRACT` — `ComputePool` deployment address
//! - `CITRATE_POOL_MEMBER_ENDPOINTS` — `addr1=url1,addr2=url2,...`
//!
//! # Optional env vars
//!
//! - `CITRATE_POOL_CHAIN_ID` — default 40204 (testnet)
//! - `CITRATE_POOL_RPC_URL` — default http://127.0.0.1:8545
//! - `CITRATE_POOL_PROVIDER_TIMEOUT_SECS` — default 30
//! - `CITRATE_POOL_POLL_INTERVAL_SECS` — event polling cadence,
//!   default 3
//! - `CITRATE_POOL_FROM_BLOCK` — initial block to poll from,
//!   default "latest" (start at the current tip; historical
//!   events are ignored)

use std::collections::{HashSet, VecDeque};
use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use ethereum_types::{H160, H256};
use tokio::sync::Mutex;
use tracing_subscriber::{fmt, EnvFilter};

use citrate_pool_coordinator::{
    handle_event, log_identity, metrics, CoordinatorConfig, CoordinatorError, HttpChainAdapter,
    Wallet,
};

/// Maximum number of (tx_hash, log_index) entries the dedup set
/// tracks. Entries older than the current `last_block -
/// confirmations_buffer` age out naturally; this cap is a
/// memory-safety bound for chains with unexpectedly long reorg
/// tails or misconfigured buffer values.
const SEEN_EVENTS_CAP: usize = 10_000;

/// CP-B-013: a FIFO-ordered dedup set for `(tx_hash, log_index)` event
/// keys. The previous implementation used a bare `HashSet` and, when
/// capped, evicted `set.iter().take(n)` — but `HashSet` iteration order
/// is `RandomState`-randomised, so it dropped an *arbitrary* half rather
/// than the "oldest half" the comment claimed. A still-recent key
/// evicted while inside the reorg re-scan window would then be
/// re-dispatched. A `VecDeque` insertion-order queue alongside the
/// membership set evicts the genuinely-oldest entries, matching the
/// `ReplayGuard` discipline already used in `libp2p_transport`.
struct SeenEvents {
    set: HashSet<(H256, u32)>,
    order: VecDeque<(H256, u32)>,
    cap: usize,
}

impl SeenEvents {
    fn with_cap(cap: usize) -> Self {
        Self {
            set: HashSet::new(),
            order: VecDeque::new(),
            cap,
        }
    }

    fn len(&self) -> usize {
        self.set.len()
    }

    /// Insert `key`. Returns `true` if it was new (not a replay). Evicts
    /// the oldest entries first when at capacity.
    fn insert(&mut self, key: (H256, u32)) -> bool {
        if self.set.contains(&key) {
            return false;
        }
        while self.set.len() >= self.cap {
            match self.order.pop_front() {
                Some(oldest) => {
                    self.set.remove(&oldest);
                }
                None => break,
            }
        }
        self.order.push_back(key);
        self.set.insert(key)
    }
}

/// Load the wallet before the Tokio runtime (and its worker threads) exists:
/// `Wallet::from_env` removes the secret from the environment, and
/// `std::env::remove_var` is only sound while the process is single-threaded
/// (PBA-L4-010).
fn main() -> ExitCode {
    let wallet = Wallet::from_env();
    match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt.block_on(async_main(wallet)),
        Err(e) => {
            eprintln!("could not start the async runtime: {e}");
            ExitCode::from(1)
        }
    }
}

async fn async_main(
    wallet: Result<Wallet, citrate_pool_coordinator::wallet::WalletError>,
) -> ExitCode {
    init_tracing();

    let cfg = match CoordinatorConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            print_config_help(&e);
            return ExitCode::from(1);
        }
    };

    let wallet = match wallet {
        Ok(w) => w,
        Err(e) => {
            eprintln!("wallet load failed: {}", e);
            eprintln!();
            eprintln!("Set one of:");
            eprintln!(
                "  CITRATE_POOL_KEYSTORE_PATH + CITRATE_POOL_KEYSTORE_PASSPHRASE  (production)"
            );
            eprintln!(
                "  CITRATE_POOL_PRIVATE_KEY_HEX                                   (testnet only)"
            );
            return ExitCode::from(2);
        }
    };

    if let Err(e) = wallet.verify_address(cfg.wallet_address) {
        eprintln!("wallet key/address mismatch: {}", e);
        return ExitCode::from(3);
    }

    let pool_contract = match env_address("CITRATE_POOL_CONTRACT") {
        Ok(a) => a,
        Err(e) => {
            eprintln!("CITRATE_POOL_CONTRACT: {}", e);
            return ExitCode::from(4);
        }
    };

    // FWA-C8-01: try_new re-validates rpc_url through the outbound TLS
    // gate at the adapter boundary (defense-in-depth behind the
    // CoordinatorConfig::from_env check). Plaintext-remote RPC dies
    // fail-closed here rather than silently MITM-able mid-run.
    let adapter = match HttpChainAdapter::try_new(
        cfg.rpc_url.clone(),
        cfg.chain_id,
        pool_contract,
        wallet.clone(),
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("RPC endpoint rejected: {}", e);
            return ExitCode::from(7);
        }
    };

    log_identity(cfg.wallet_address, cfg.member_endpoints.len());
    tracing::info!(
        rpc_url = %cfg.rpc_url,
        chain_id = cfg.chain_id,
        pool_contract = ?pool_contract,
        "daemon configured"
    );

    // RM-B1 / WP-B2.4 (audit F-5): fail-fast if the RPC endpoint's
    // advertised chain_id doesn't match the configured one. Pre-fix,
    // a misconfigured daemon could sign transactions for the wrong
    // chain and silently fail until an operator noticed.
    if let Err(e) = adapter.verify_rpc_chain_id().await {
        eprintln!("F-5 chain_id verification failed: {}", e);
        return ExitCode::from(6);
    }
    tracing::info!(chain_id = cfg.chain_id, "F-5: RPC chain_id verified");

    // Spawn the Prometheus metrics server if configured.
    if let Ok(bind) = env::var("CITRATE_POOL_METRICS_ADDR") {
        if let Err(e) = metrics::spawn_metrics_server(&bind).await {
            tracing::warn!(bind = %bind, error = %e, "metrics server failed to start");
        }
    }

    // Determine starting block. Default: current tip (skip history).
    let mut last_block = match starting_block(&adapter).await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to fetch starting block: {}", e);
            return ExitCode::from(5);
        }
    };
    tracing::info!(starting_block = last_block, "event polling starts here");

    let poll_interval = Duration::from_secs(
        env::var("CITRATE_POOL_POLL_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
    );

    // Reorg tolerance: re-scan `[last_block - buffer, latest]` each
    // tick, deduping seen events by (tx_hash, log_index). A chain
    // re-org within `buffer` blocks may surface NEW events at
    // previously-seen block numbers or drop events we already
    // dispatched; the dedup set prevents re-dispatch of stable
    // events and lets genuinely new events through.
    let confirmations_buffer: u64 = env::var("CITRATE_POOL_CONFIRMATIONS_BUFFER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let seen: Arc<Mutex<SeenEvents>> = Arc::new(Mutex::new(SeenEvents::with_cap(SEEN_EVENTS_CAP)));

    // Event polling loop.
    //
    // Per tick: fetch latest block, poll [max(0, last_block - buffer),
    // latest] for ComputeRequested, dedupe seen events, dispatch
    // fresh ones to handle_event in spawned tasks, advance
    // last_block to latest.
    //
    // Safety properties this design maintains:
    // - Each event dispatches at most once per daemon run (dedup
    //   set keyed on (tx_hash, log_index)).
    // - Chain reorgs within `buffer` blocks surface new events
    //   automatically on the next tick without manual intervention.
    // - Slow handle_event runs don't delay event discovery.
    // - Backpressure: dedup set capped at SEEN_EVENTS_CAP; beyond
    //   that we clear half to bound memory.
    loop {
        match tick(&adapter, &cfg, last_block, confirmations_buffer, &seen).await {
            Ok(new_last) => last_block = new_last,
            Err(e) => {
                ::metrics::counter!("pool_coord_poll_failures_total").increment(1);
                tracing::warn!(error = %e, "poll tick failed; retrying after interval");
            }
        }
        // Update gauge after each tick.
        {
            let set = seen.lock().await;
            ::metrics::gauge!("pool_coord_seen_events_cardinality").set(set.len() as f64);
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn tick(
    adapter: &HttpChainAdapter,
    cfg: &CoordinatorConfig,
    last_block: u64,
    confirmations_buffer: u64,
    seen: &Arc<Mutex<SeenEvents>>,
) -> Result<u64, CoordinatorError> {
    let latest = adapter.latest_block().await?;
    if latest <= last_block {
        return Ok(last_block);
    }
    // Widen the window backwards by `confirmations_buffer` so reorg
    // backfills are observed. Floor at 0.
    let from = last_block.saturating_sub(confirmations_buffer).max(1);
    let events = adapter.poll_compute_requested(from, latest).await?;

    // CP-B-013: eviction is FIFO (oldest-first) inside `SeenEvents::insert`,
    // so a still-recent key can no longer be dropped ahead of an older one.

    for event in events {
        ::metrics::counter!("pool_coord_events_observed_total").increment(1);
        // Dedup: skip if we've already dispatched this (tx_hash, log_index).
        {
            let mut set = seen.lock().await;
            let key = (event.tx_hash, event.log_index);
            if !set.insert(key) {
                continue;
            }
        }
        let cfg_clone = cfg.clone();
        let adapter_clone = adapter.clone();
        tokio::spawn(async move {
            match handle_event(&adapter_clone, &cfg_clone, &event).await {
                Ok(_) => {
                    ::metrics::counter!(
                        "pool_coord_events_dispatched_total",
                        "outcome" => "success"
                    )
                    .increment(1);
                    tracing::info!(pool = event.pool_id, job = event.job_id, "event handled");
                }
                Err(CoordinatorError::NotCoordinator) => {
                    ::metrics::counter!(
                        "pool_coord_events_dispatched_total",
                        "outcome" => "not_coord"
                    )
                    .increment(1);
                    tracing::debug!(
                        pool = event.pool_id,
                        job = event.job_id,
                        "skipped (not coordinator)"
                    );
                }
                Err(e) => {
                    ::metrics::counter!(
                        "pool_coord_events_dispatched_total",
                        "outcome" => "error"
                    )
                    .increment(1);
                    tracing::warn!(
                        pool = event.pool_id,
                        job = event.job_id,
                        error = %e,
                        "handle_event failed"
                    );
                }
            }
        });
    }
    Ok(latest)
}

async fn starting_block(adapter: &HttpChainAdapter) -> Result<u64, CoordinatorError> {
    if let Ok(s) = env::var("CITRATE_POOL_FROM_BLOCK") {
        if let Ok(n) = s.parse::<u64>() {
            return Ok(n);
        }
        if s == "latest" {
            return adapter.latest_block().await;
        }
    }
    adapter.latest_block().await
}

fn env_address(key: &str) -> Result<H160, String> {
    let raw = env::var(key).map_err(|_| format!("{} unset", key))?;
    let trimmed = raw.trim().trim_start_matches("0x");
    if trimmed.len() != 40 {
        return Err(format!("{} wants 40 hex chars, got {}", key, trimmed.len()));
    }
    let bytes = hex::decode(trimmed).map_err(|e| format!("{} bad hex: {}", key, e))?;
    let mut arr = [0u8; 20];
    arr.copy_from_slice(&bytes);
    Ok(H160::from(arr))
}

fn init_tracing() {
    let format = env::var("LOG_FORMAT").unwrap_or_else(|_| "pretty".into());
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,citrate_pool_coordinator=debug"));
    if format == "json" {
        fmt().with_env_filter(filter).json().init();
    } else {
        fmt().with_env_filter(filter).init();
    }
}

fn print_config_help(err: &str) {
    eprintln!("config error: {}", err);
    eprintln!();
    eprintln!("Required env vars:");
    eprintln!("  Wallet (one of):");
    eprintln!("    CITRATE_POOL_KEYSTORE_PATH        Web3 SSv3 keystore file");
    eprintln!("    CITRATE_POOL_KEYSTORE_PASSPHRASE  passphrase for the keystore");
    eprintln!("   OR");
    eprintln!("    CITRATE_POOL_PRIVATE_KEY_HEX      64 hex chars (testnet only)");
    eprintln!("  CITRATE_POOL_WALLET_ADDRESS   0x + 40 hex (must match derived)");
    eprintln!("  CITRATE_POOL_CONTRACT         0x + 40 hex (ComputePool deploy)");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  CITRATE_POOL_CHAIN_ID             default 40204");
    eprintln!("  CITRATE_POOL_RPC_URL              default http://127.0.0.1:8545");
    eprintln!("  CITRATE_POOL_MEMBER_ENDPOINTS     addr1=url1,addr2=url2");
    eprintln!("  CITRATE_POOL_PROVIDER_TIMEOUT_SECS  default 30");
    eprintln!("  CITRATE_POOL_POLL_INTERVAL_SECS   default 3");
    eprintln!("  CITRATE_POOL_FROM_BLOCK           default \"latest\"");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> (H256, u32) {
        (H256::from([n; 32]), n as u32)
    }

    // CP-B-013 tripwire: at capacity the dedup set must evict the
    // OLDEST entries, so the most-recently-inserted keys survive and
    // cannot be re-dispatched. Pre-fix (HashSet::iter().take) this was
    // non-deterministic and could evict a still-recent key.
    #[test]
    fn seen_events_evicts_oldest_first_and_keeps_recent() {
        let mut seen = SeenEvents::with_cap(3);
        assert!(seen.insert(key(1)));
        assert!(seen.insert(key(2)));
        assert!(seen.insert(key(3)));
        // Inserting a 4th at cap must evict the oldest (key 1), never a
        // more-recent one.
        assert!(seen.insert(key(4)));
        assert_eq!(seen.len(), 3);
        // The three most-recent keys survive → replays are still caught.
        assert!(!seen.insert(key(2)), "recent key 2 must still be deduped");
        assert!(!seen.insert(key(3)), "recent key 3 must still be deduped");
        assert!(!seen.insert(key(4)), "recent key 4 must still be deduped");
        // The evicted oldest key is treated as new (acceptable: it has
        // aged out of the reorg window).
        assert!(seen.insert(key(1)));
    }

    #[test]
    fn seen_events_dedups_exact_replay() {
        let mut seen = SeenEvents::with_cap(10);
        assert!(seen.insert(key(7)));
        assert!(!seen.insert(key(7)), "exact replay must be refused");
        assert_eq!(seen.len(), 1);
    }
}
