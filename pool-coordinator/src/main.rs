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
//! - `CITRATE_POOL_RPC_URL` — default http://127.0.0.1:18545
//! - `CITRATE_POOL_PROVIDER_TIMEOUT_SECS` — default 30
//! - `CITRATE_POOL_POLL_INTERVAL_SECS` — event polling cadence,
//!   default 3
//! - `CITRATE_POOL_FROM_BLOCK` — initial block to poll from,
//!   default "latest" (start at the current tip; historical
//!   events are ignored)

use std::env;
use std::process::ExitCode;
use std::time::Duration;

use ethereum_types::H160;
use tracing_subscriber::{fmt, EnvFilter};

use citrate_pool_coordinator::{
    handle_event, log_identity, CoordinatorConfig, CoordinatorError, HttpChainAdapter, Wallet,
};

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();

    let cfg = match CoordinatorConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            print_config_help(&e);
            return ExitCode::from(1);
        }
    };

    let wallet = match Wallet::from_env() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("wallet load failed: {}", e);
            eprintln!();
            eprintln!("Set one of:");
            eprintln!("  CITRATE_POOL_KEYSTORE_PATH + CITRATE_POOL_KEYSTORE_PASSPHRASE  (production)");
            eprintln!("  CITRATE_POOL_PRIVATE_KEY_HEX                                   (testnet only)");
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

    let adapter = HttpChainAdapter::new(
        cfg.rpc_url.clone(),
        cfg.chain_id,
        pool_contract,
        wallet.clone(),
    );

    log_identity(cfg.wallet_address, cfg.member_endpoints.len());
    tracing::info!(
        rpc_url = %cfg.rpc_url,
        chain_id = cfg.chain_id,
        pool_contract = ?pool_contract,
        "daemon configured"
    );

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

    // Event polling loop.
    //
    // Per tick: fetch latest block, poll [last_block+1, latest]
    // for ComputeRequested, dispatch each event to handle_event
    // in a spawned task (so one slow dispatch doesn't block the
    // next poll), advance last_block to latest.
    //
    // Safety properties this design maintains:
    // - Each event dispatches at most once per daemon run (we
    //   advance last_block only after the poll completes).
    // - Slow handle_event runs don't delay event discovery.
    // - Backpressure: if handle_event panics (it shouldn't —
    //   CoordinatorError is the error surface), the daemon keeps
    //   polling; tokio reports the join failure.
    loop {
        match tick(&adapter, &cfg, last_block, &wallet).await {
            Ok(new_last) => last_block = new_last,
            Err(e) => {
                tracing::warn!(error = %e, "poll tick failed; retrying after interval");
            }
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn tick(
    adapter: &HttpChainAdapter,
    cfg: &CoordinatorConfig,
    last_block: u64,
    _wallet: &Wallet,
) -> Result<u64, CoordinatorError> {
    let latest = adapter.latest_block().await?;
    if latest <= last_block {
        return Ok(last_block);
    }
    let from = last_block + 1;
    let events = adapter.poll_compute_requested(from, latest).await?;
    for event in events {
        let cfg_clone = cfg.clone();
        let adapter_clone = adapter.clone();
        tokio::spawn(async move {
            match handle_event(&adapter_clone, &cfg_clone, &event).await {
                Ok(_) => tracing::info!(
                    pool = event.pool_id,
                    job = event.job_id,
                    "event handled"
                ),
                Err(CoordinatorError::NotCoordinator) => {
                    // Expected for events where another peer is the
                    // coordinator. Noise-suppressed at info level.
                    tracing::debug!(
                        pool = event.pool_id,
                        job = event.job_id,
                        "skipped (not coordinator)"
                    );
                }
                Err(e) => tracing::warn!(
                    pool = event.pool_id,
                    job = event.job_id,
                    error = %e,
                    "handle_event failed"
                ),
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
        return Err(format!(
            "{} wants 40 hex chars, got {}",
            key,
            trimmed.len()
        ));
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
    eprintln!("  CITRATE_POOL_RPC_URL              default http://127.0.0.1:18545");
    eprintln!("  CITRATE_POOL_MEMBER_ENDPOINTS     addr1=url1,addr2=url2");
    eprintln!("  CITRATE_POOL_PROVIDER_TIMEOUT_SECS  default 30");
    eprintln!("  CITRATE_POOL_POLL_INTERVAL_SECS   default 3");
    eprintln!("  CITRATE_POOL_FROM_BLOCK           default \"latest\"");
}
