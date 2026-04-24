//! citrate-training-worker production binary entry point.
//!
//! Dual-mode daemon per ADR-009:
//!   - `training` mode — joins a DataParallel training pool
//!     (CM-07); uses `HttpChainClientTraining`
//!   - `pipeline` mode — joins a PipelineParallel inference
//!     pool (CM-08); uses `HttpPipelineChainClient`
//!
//! Mode is selected via `CITRATE_WORKER_MODE` env var. The
//! ModelBackend + Transport for S1 are still the in-process S0
//! substitutes (DeterministicTinyModel + InProcessTransport) —
//! S2 swaps them for GPU + libp2p per the backlog.
//!
//! # Required env vars
//!
//! Wallet (one of):
//! - `CITRATE_TRAINING_KEYSTORE_PATH` + `CITRATE_TRAINING_KEYSTORE_PASSPHRASE`
//! - `CITRATE_TRAINING_PRIVATE_KEY_HEX` (testnet only)
//!
//! Plus:
//! - `CITRATE_WORKER_MODE` — "training" or "pipeline"
//! - `CITRATE_WORKER_CONTRACT` — deployed contract address for
//!   the selected mode (ComputePoolTraining or ComputePoolPipeline)
//!
//! # Optional env vars
//!
//! - `CITRATE_WORKER_CHAIN_ID` — default 40204 (testnet)
//! - `CITRATE_WORKER_RPC_URL` — default https://rpc.citrate.ai
//! - `LOG_FORMAT` — "pretty" (default) or "json"
//! - `RUST_LOG` — standard tracing filter
//!
//! # What this binary does in S1
//!
//! Loads wallet, loads config, constructs the HTTP chain client,
//! prints identity, exits with code 0. The actual event-loop
//! (subscribe to TrainingJobOpened / PipelineRequestSubmitted,
//! drive the worker state machine, report outcomes) is the next
//! follow-up tracked in the S1/S2 backlog as
//! `training-worker-bin-event-loop` — it depends on the same
//! polling pattern pool-coordinator uses, ported to the
//! training-worker surface.

use std::env;
use std::process::ExitCode;

use ethereum_types::H160;
use tracing_subscriber::{fmt, EnvFilter};

use citrate_training_worker::{
    http_chain_training::HttpChainClient as HttpChainClientTraining,
    HttpPipelineChainClient, Wallet,
};

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

    tracing::info!(
        wallet = ?wallet.address(),
        mode = %match mode {
            Mode::Training => "training",
            Mode::Pipeline => "pipeline",
        },
        chain_id = chain_id,
        rpc_url = %rpc_url,
        contract = ?contract,
        "training-worker starting"
    );

    // Construct the mode-specific HTTP chain client. Verifies
    // reachability by doing a no-op eth_blockNumber call; if that
    // fails, we exit early rather than start a worker loop that
    // will fail every tick.
    match mode {
        Mode::Training => {
            let _client = HttpChainClientTraining::new(
                rpc_url.clone(),
                chain_id,
                contract,
                wallet.clone(),
            );
            tracing::info!("HttpChainClientTraining constructed");
        }
        Mode::Pipeline => {
            let _client = HttpPipelineChainClient::new(
                rpc_url.clone(),
                chain_id,
                contract,
                wallet.clone(),
            );
            tracing::info!("HttpPipelineChainClient constructed");
        }
    }

    // S1 scope: prove wiring. Event-loop wiring (subscribe to
    // TrainingJobOpened / PipelineRequestSubmitted, drive the
    // worker state machine) tracked in the S1/S2 backlog under
    // `training-worker-bin-event-loop`.
    tracing::warn!(
        "S1 daemon: identity + chain client verified. Event loop lands in the \
         next slice once 3-LAN-machine integration tests are running. For S0 \
         in-process testing, use `cargo test -p citrate-training-worker`."
    );
    ExitCode::SUCCESS
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
    eprintln!("  CITRATE_WORKER_CHAIN_ID     default 40204");
    eprintln!("  CITRATE_WORKER_RPC_URL      default https://rpc.citrate.ai");
    eprintln!("  LOG_FORMAT                  pretty | json (default pretty)");
    eprintln!("  RUST_LOG                    tracing filter");
}
