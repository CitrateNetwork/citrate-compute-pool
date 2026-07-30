//! `citrate-coop-worker` — the member-facing daemon.
//!
//! What a co-op member actually runs. It registers the machine with the training
//! coordinator using a measured probe, then polls for work until stopped.
//!
//! ```sh
//! # 1. measure the machine (in the nat checkout)
//! cargo run --release -p nat-candle --example divergence_probe > probe.json
//!
//! # 2. join
//! CITRATE_COORDINATOR_URL=https://coordinator.citrate.ai \
//! CITRATE_PROBE_PATH=./probe.json \
//! CITRATE_TRAINING_KEYSTORE_PATH=~/.citrate/worker.json \
//! CITRATE_TRAINING_KEYSTORE_PASSPHRASE=... \
//!   citrate-coop-worker
//! ```
//!
//! There is no API key. The keystore the worker already uses is the identity: the
//! coordinator recovers the signing address from each request, so a member has
//! one key and nothing to rotate.
//!
//! # Scope, stated plainly
//!
//! **This binary does not execute training jobs yet.** It registers, polls, and
//! reports what it is asked to do. Any job it is offered is refused with a
//! reason, its lease expires, and the coordinator hands the work to another
//! machine — which is the correct behaviour for a worker that cannot do the job,
//! and is exactly the `failed_by` path the coordinator implements.
//!
//! That is a deliberate boundary, not an oversight. Executing a job means turning
//! a `JobSpec` payload into a real run against `NatBackend` with verified
//! artifacts, and a worker that *pretended* to complete jobs would poison the
//! ladder with fabricated results — far worse than one that honestly declines.
//! [`JobRunner`] is the seam that work plugs into.
//!
//! What it IS useful for today: proving a member's machine can reach the
//! coordinator, that its keystore signs correctly, and that its measured
//! capability is what they expect — before any GPU time is committed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use citrate_training_worker::coordinator_client::CoordinatorClient;
use citrate_training_worker::coordinator_protocol::JobSpec;
use citrate_training_worker::wallet::Wallet;

/// How a job gets executed. The insertion point for the training backend.
#[allow(async_fn_in_trait)]
pub trait JobRunner {
    /// Return the payload to submit, or an error to decline the job.
    async fn run(&self, job: JobSpec) -> anyhow::Result<String>;
}

/// The runner this binary ships with: it declines everything, loudly and with a
/// reason. Named for what it does so no one reads it as a stub that "works".
struct DeclineUntilBackendWired;

impl JobRunner for DeclineUntilBackendWired {
    async fn run(&self, job: JobSpec) -> anyhow::Result<String> {
        anyhow::bail!(
            "this worker cannot execute job {} ({:?}): the training backend is not \
             wired to the coordinator yet. Declining so the lease expires and the \
             coordinator reassigns it.",
            job.id,
            job.requires
        )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let url = std::env::var("CITRATE_COORDINATOR_URL")
        .map_err(|_| anyhow::anyhow!("CITRATE_COORDINATOR_URL is required"))?;
    let probe_path = std::env::var("CITRATE_PROBE_PATH").unwrap_or_else(|_| "./probe.json".into());

    let probe = std::fs::read_to_string(&probe_path).map_err(|e| {
        anyhow::anyhow!(
            "could not read the probe at {probe_path}: {e}. Produce one with: \
             cargo run --release -p nat-candle --example divergence_probe > probe.json"
        )
    })?;

    // Same key the worker transacts with, so a member has ONE identity.
    let wallet = Wallet::from_env()?;
    let client = CoordinatorClient::new(&url, wallet);
    tracing::info!(worker = ?client.worker_id(), coordinator = %url, "starting");

    // Registration is also the connectivity check: if this succeeds, the member
    // knows their key signs correctly and what their machine is rated for.
    let reg = client.register(&probe).await?;
    tracing::info!(
        capability = ?reg.capability,
        worker = %reg.worker,
        "registered — the coordinator rated this machine from the probe"
    );

    // Ctrl-C stops after the current poll rather than mid-job.
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("stopping after the current poll");
            r.store(false, Ordering::SeqCst);
        }
    });

    let runner = Arc::new(DeclineUntilBackendWired);
    let check = running.clone();
    client
        .poll_loop(
            move |job| {
                let r = runner.clone();
                async move { r.run(job).await }
            },
            move || check.load(Ordering::SeqCst),
        )
        .await?;

    tracing::info!("stopped");
    Ok(())
}
