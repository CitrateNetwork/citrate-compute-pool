//! `citrate-pool-coordinator` — binary entry point.
//!
//! Slice 1: this binary loads its config, prints its identity, and
//! exits with a clear "slice 2 wires the live event subscription"
//! message. The library's `handle_event` is the testable core; the
//! glue between an `eth_subscribe newHeads` stream and that function
//! lands in slice 2 alongside the secp256k1 wallet for tx signing.

use std::env;
use std::process::ExitCode;

use citrate_pool_coordinator::{log_identity, CoordinatorConfig};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> ExitCode {
    let format = env::var("LOG_FORMAT").unwrap_or_else(|_| "pretty".to_string());
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,citrate_pool_coordinator=debug"));
    if format == "json" {
        fmt().with_env_filter(filter).json().init();
    } else {
        fmt().with_env_filter(filter).init();
    }

    let cfg = match CoordinatorConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {}", e);
            eprintln!();
            eprintln!("Required env vars:");
            eprintln!("  CITRATE_POOL_WALLET_ADDRESS  0x + 40 hex");
            eprintln!();
            eprintln!("Optional:");
            eprintln!("  CITRATE_POOL_CHAIN_ID        default 40204");
            eprintln!("  CITRATE_POOL_RPC_URL         default http://127.0.0.1:18545");
            eprintln!("  CITRATE_POOL_MEMBER_ENDPOINTS  addr1=url1,addr2=url2");
            eprintln!("  CITRATE_POOL_PROVIDER_TIMEOUT_SECS  default 30");
            return ExitCode::from(1);
        }
    };

    log_identity(cfg.wallet_address, cfg.member_endpoints.len());

    tracing::warn!(
        "slice 1 binary: live event subscription + tx signing land in WP-05.2 slice 2. \
         Use the library entry point `handle_event` for integration tests today."
    );
    ExitCode::SUCCESS
}
