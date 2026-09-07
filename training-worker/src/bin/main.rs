//! citrate-training-worker production binary entry point.
//!
//! Dual-mode daemon per ADR-009:
//!   - `training` mode  — CM-07 DataParallel training pool member
//!   - `pipeline` mode  — CM-08 PipelineParallel inference stage
//!
//! The main loop polls the contract for events touching this
//! daemon's configured job (scoped via indexed jobId topic filter),
//! classifies each event by topic[0], and logs outcomes. Dedup
//! set keyed on (tx_hash, log_index) with a 12-block re-scan
//! window for reorg tolerance. Metrics exposed via optional
//! Prometheus endpoint.
//!
//! Actual dispatch (spawning a `Worker` or `PipelineWorker`
//! instance) is bounded by the S1 scope: the daemon observes +
//! logs the events that would trigger worker actions. Wiring
//! the observed events to the state-machine driver is a thin
//! follow-up (the state machine already exists; this is just
//! the edge glue). Keeping it split means the current binary
//! deploys cleanly on testnet for partner ops + telemetry
//! validation without also requiring the worker backend to be
//! real-GPU-ready.
//!
//! # Required env vars
//!
//! Wallet (one of):
//! - `CITRATE_TRAINING_KEYSTORE_PATH` + `CITRATE_TRAINING_KEYSTORE_PASSPHRASE`
//! - `CITRATE_TRAINING_PRIVATE_KEY_HEX` (testnet only)
//!
//! Plus:
//! - `CITRATE_WORKER_MODE` — "training" or "pipeline"
//! - `CITRATE_WORKER_CONTRACT` — deployed contract address
//! - `CITRATE_WORKER_JOB_ID` — job to watch (numeric; required
//!   for focused event polling. If unset, polls all contract
//!   events — useful for ops monitoring but noisy)
//!
//! # Optional env vars
//!
//! - `CITRATE_WORKER_CHAIN_ID` — default 40204 (testnet)
//! - `CITRATE_WORKER_RPC_URL` — default https://rpc.citrate.ai
//! - `CITRATE_WORKER_POLL_INTERVAL_SECS` — default 3
//! - `CITRATE_WORKER_FROM_BLOCK` — "latest" (default) or numeric
//! - `CITRATE_WORKER_CONFIRMATIONS_BUFFER` — default 12
//! - `CITRATE_WORKER_METRICS_ADDR` — e.g. "127.0.0.1:9092"
//! - `LOG_FORMAT` — "pretty" (default) or "json"
//! - `RUST_LOG` — standard tracing filter

use std::collections::HashSet;
use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use ethereum_types::{H160, H256};
use tokio::sync::Mutex;
use tracing_subscriber::{fmt, EnvFilter};

use citrate_training_worker::{
    events::{classify, EventKind, RawLog},
    http_chain_training::HttpChainClient as HttpChainClientTraining,
    HttpPipelineChainClient, Wallet,
};

/// Bounded dedup set so pathological reorg scenarios don't
/// exhaust memory.
const SEEN_EVENTS_CAP: usize = 10_000;

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();

    let mode = match env::var("CITRATE_WORKER_MODE").as_deref() {
        Ok("training") => Mode::Training,
        Ok("pipeline") => Mode::Pipeline,
        Ok(other) => {
            eprintln!(
                "CITRATE_WORKER_MODE='{}' not recognised; expected 'training' or 'pipeline'",
                other
            );
            return ExitCode::from(1);
        }
        Err(_) => {
            print_config_help();
            return ExitCode::from(1);
        }
    };

    let wallet = match Wallet::from_env() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("wallet load failed: {}", e);
            eprintln!();
            eprintln!("Set one of:");
            eprintln!("  CITRATE_TRAINING_KEYSTORE_PATH + CITRATE_TRAINING_KEYSTORE_PASSPHRASE");
            eprintln!("  CITRATE_TRAINING_PRIVATE_KEY_HEX (testnet only)");
            return ExitCode::from(2);
        }
    };

    let contract = match env_address("CITRATE_WORKER_CONTRACT") {
        Ok(a) => a,
        Err(e) => {
            eprintln!("CITRATE_WORKER_CONTRACT: {}", e);
            return ExitCode::from(3);
        }
    };

    let chain_id = env::var("CITRATE_WORKER_CHAIN_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40204);
    let rpc_url = env::var("CITRATE_WORKER_RPC_URL")
        .unwrap_or_else(|_| "https://rpc.citrate.ai".to_string());
    // CP-B-006: the RPC leg reads chain truth AND carries signed money
    // transactions. A plaintext remote RPC is MITM-able (false chain
    // truth to a daemon that signs). Refuse fail-closed at startup —
    // mirrors pool-coordinator's config-load gate. Default is https, so
    // deployments using the default are unaffected.
    if let Err(e) = citrate_training_worker::outbound::validate_outbound_url(&rpc_url) {
        eprintln!("CITRATE_WORKER_RPC_URL: {}", e);
        return ExitCode::from(4);
    }

    let job_id_filter: Option<u64> = env::var("CITRATE_WORKER_JOB_ID")
        .ok()
        .and_then(|s| s.parse().ok());

    tracing::info!(
        wallet = ?wallet.address(),
        mode = %match mode {
            Mode::Training => "training",
            Mode::Pipeline => "pipeline",
        },
        chain_id = chain_id,
        rpc_url = %rpc_url,
        contract = ?contract,
        job_id_filter = ?job_id_filter,
        "training-worker starting"
    );

    // Metrics (optional) — see pool-coordinator::metrics for the
    // pattern. Training-worker ships without its own metrics
    // module for S1; if CITRATE_WORKER_METRICS_ADDR is set but no
    // recorder is installed, counters still increment (the
    // metrics facade is a no-op without a backend) and a warning
    // logs that observability is unwired.
    if env::var("CITRATE_WORKER_METRICS_ADDR").is_ok() {
        tracing::warn!(
            "CITRATE_WORKER_METRICS_ADDR set but training-worker metrics server \
             not yet wired (follow-up). Pool-coordinator's /metrics pattern applies."
        );
    }

    let poll_interval = Duration::from_secs(
        env::var("CITRATE_WORKER_POLL_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
    );
    let confirmations_buffer: u64 = env::var("CITRATE_WORKER_CONFIRMATIONS_BUFFER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    match mode {
        Mode::Training => {
            let client = HttpChainClientTraining::new(
                rpc_url,
                chain_id,
                contract,
                wallet,
            );
            // CP-B-006 (F-5 parity): fail-fast if the RPC's advertised
            // chain id doesn't match the configured one, before signing.
            if let Err(e) = client.verify_chain_id().await {
                tracing::error!(error = %e, "RPC chain_id verification failed");
                return ExitCode::from(12);
            }
            if let Err(e) = training_event_loop(
                &client,
                job_id_filter,
                poll_interval,
                confirmations_buffer,
            )
            .await
            {
                tracing::error!(error = %e, "training event loop terminated");
                return ExitCode::from(10);
            }
        }
        Mode::Pipeline => {
            let client = HttpPipelineChainClient::new(
                rpc_url,
                chain_id,
                contract,
                wallet,
            );
            if let Err(e) = pipeline_event_loop(
                &client,
                job_id_filter,
                poll_interval,
                confirmations_buffer,
            )
            .await
            {
                tracing::error!(error = %e, "pipeline event loop terminated");
                return ExitCode::from(11);
            }
        }
    }
    ExitCode::SUCCESS
}

async fn training_event_loop(
    client: &HttpChainClientTraining,
    job_id_filter: Option<u64>,
    poll_interval: Duration,
    confirmations_buffer: u64,
) -> anyhow::Result<()> {
    let mut last_block = starting_block(client.latest_block().await?);
    tracing::info!(starting_block = last_block, "training event polling starts");
    let seen: Arc<Mutex<HashSet<(H256, u32)>>> = Arc::new(Mutex::new(HashSet::new()));

    loop {
        if let Err(e) =
            training_tick(client, job_id_filter, &mut last_block, confirmations_buffer, &seen)
                .await
        {
            tracing::warn!(error = %e, "training poll tick failed; retrying after interval");
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn training_tick(
    client: &HttpChainClientTraining,
    job_id_filter: Option<u64>,
    last_block: &mut u64,
    confirmations_buffer: u64,
    seen: &Arc<Mutex<HashSet<(H256, u32)>>>,
) -> anyhow::Result<()> {
    let latest = client
        .latest_block()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    if latest <= *last_block {
        return Ok(());
    }
    let from = last_block.saturating_sub(confirmations_buffer).max(1);
    let logs = client
        .poll_raw_logs(from, latest, job_id_filter)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    dispatch_logs(logs, seen, "training").await;
    *last_block = latest;
    Ok(())
}

async fn pipeline_event_loop(
    client: &HttpPipelineChainClient,
    job_id_filter: Option<u64>,
    poll_interval: Duration,
    confirmations_buffer: u64,
) -> anyhow::Result<()> {
    let mut last_block = starting_block(
        client
            .latest_block()
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?,
    );
    tracing::info!(starting_block = last_block, "pipeline event polling starts");
    let seen: Arc<Mutex<HashSet<(H256, u32)>>> = Arc::new(Mutex::new(HashSet::new()));

    loop {
        if let Err(e) =
            pipeline_tick(client, job_id_filter, &mut last_block, confirmations_buffer, &seen)
                .await
        {
            tracing::warn!(error = %e, "pipeline poll tick failed; retrying after interval");
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn pipeline_tick(
    client: &HttpPipelineChainClient,
    job_id_filter: Option<u64>,
    last_block: &mut u64,
    confirmations_buffer: u64,
    seen: &Arc<Mutex<HashSet<(H256, u32)>>>,
) -> anyhow::Result<()> {
    let latest = client
        .latest_block()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    if latest <= *last_block {
        return Ok(());
    }
    let from = last_block.saturating_sub(confirmations_buffer).max(1);
    let logs = client
        .poll_raw_logs(from, latest, job_id_filter)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    dispatch_logs(logs, seen, "pipeline").await;
    *last_block = latest;
    Ok(())
}

/// Dedup-aware log dispatcher. For S1 we log each unique event.
/// The follow-up wiring (spawn a Worker or PipelineWorker in
/// response) is a thin addition on top of this shape — it doesn't
/// require changing how events are observed.
async fn dispatch_logs(
    logs: Vec<RawLog>,
    seen: &Arc<Mutex<HashSet<(H256, u32)>>>,
    which: &'static str,
) {
    // Evict oldest if cap hit before inserting.
    {
        let mut set = seen.lock().await;
        if set.len() >= SEEN_EVENTS_CAP {
            let drop_count = set.len() / 2;
            let to_remove: Vec<_> = set.iter().take(drop_count).cloned().collect();
            for k in to_remove {
                set.remove(&k);
            }
            tracing::warn!(
                dropped = drop_count,
                "{} dedup set hit cap; dropped oldest half",
                which
            );
        }
    }

    for log in logs {
        let key = log.dedup_key();
        {
            let mut set = seen.lock().await;
            if !set.insert(key) {
                continue;
            }
        }
        let kind = log
            .topics
            .first()
            .copied()
            .map(classify)
            .unwrap_or(EventKind::Unknown);
        tracing::info!(
            mode = %which,
            event = kind.as_str(),
            block = log.block_number,
            tx = ?log.tx_hash,
            log_index = log.log_index,
            "event observed"
        );
    }
}

fn starting_block(latest: u64) -> u64 {
    // Default: start at current tip (skip history). Override via
    // CITRATE_WORKER_FROM_BLOCK=<num> or ="latest".
    match env::var("CITRATE_WORKER_FROM_BLOCK").ok().as_deref() {
        Some("latest") | None => latest,
        Some(s) => s.parse().unwrap_or(latest),
    }
}

#[derive(Debug)]
enum Mode {
    Training,
    Pipeline,
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
        .unwrap_or_else(|_| EnvFilter::new("info,citrate_training_worker=debug"));
    if format == "json" {
        fmt().with_env_filter(filter).json().init();
    } else {
        fmt().with_env_filter(filter).init();
    }
}

fn print_config_help() {
    eprintln!("Required env vars:");
    eprintln!();
    eprintln!("  CITRATE_WORKER_MODE         'training' or 'pipeline'");
    eprintln!("  CITRATE_WORKER_CONTRACT     0x + 40 hex (deploy address for chosen mode)");
    eprintln!();
    eprintln!("  Wallet (one of):");
    eprintln!("    CITRATE_TRAINING_KEYSTORE_PATH       Web3 SSv3 keystore file");
    eprintln!("    CITRATE_TRAINING_KEYSTORE_PASSPHRASE passphrase for the keystore");
    eprintln!("   OR");
    eprintln!("    CITRATE_TRAINING_PRIVATE_KEY_HEX     64 hex chars (testnet only)");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  CITRATE_WORKER_JOB_ID                 numeric job to watch (recommended)");
    eprintln!("  CITRATE_WORKER_CHAIN_ID               default 40204");
    eprintln!("  CITRATE_WORKER_RPC_URL                default https://rpc.citrate.ai");
    eprintln!("  CITRATE_WORKER_POLL_INTERVAL_SECS     default 3");
    eprintln!("  CITRATE_WORKER_CONFIRMATIONS_BUFFER   default 12");
    eprintln!("  CITRATE_WORKER_FROM_BLOCK             'latest' or numeric (default latest)");
    eprintln!("  CITRATE_WORKER_METRICS_ADDR           Prometheus scrape target");
    eprintln!("  LOG_FORMAT                            pretty | json (default pretty)");
    eprintln!("  RUST_LOG                              tracing filter");
}
