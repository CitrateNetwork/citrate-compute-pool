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
//! # Two builds, and the difference is honest
//!
//! Built **with `--features nat`**, this executes training jobs against the real
//! NAT backend via [`job_runner::NatJobRunner`] — verified artifacts, real steps,
//! a reproducible commitment.
//!
//! Built **without it**, there is no training backend compiled in, so every job
//! is declined with a reason and its lease expires for the coordinator to
//! reassign. That is not a degraded mode to apologise for: a CPU-only member
//! still registers, still contributes divergence measurements, and — critically —
//! a worker that *pretended* to complete jobs would poison the ladder with
//! fabricated results that nobody could reproduce. Declining is the correct
//! behaviour for a machine that cannot do the work.
//!
//! # Artifacts are not fetched
//!
//! With `nat`, jobs are run against a local artifact store
//! (`CITRATE_ARTIFACT_STORE`, default `./artifacts`). If the checkpoint and
//! corpus a job names are not staged there, the job is declined with the hashes
//! that were wanted. Moving 2.4 GB to volunteer machines is a distribution
//! problem, and half-solving it inside the daemon that settles money is not the
//! place to start.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use citrate_training_worker::coordinator_client::CoordinatorClient;
use citrate_training_worker::coordinator_protocol::JobSpec;
use citrate_training_worker::wallet::Wallet;

/// Executes a job, or returns the reason it will not.
///
/// Declining is a first-class outcome throughout: the coordinator lets the lease
/// expire and reassigns the work, which is why `RunError` variants explain the
/// consequence rather than just naming the fault.
struct Executor {
    #[cfg(feature = "nat")]
    inner: citrate_training_worker::job_runner::NatJobRunner,
}

impl Executor {
    #[cfg(feature = "nat")]
    fn new(worker: ethereum_types::H160) -> Self {
        let store =
            std::env::var("CITRATE_ARTIFACT_STORE").unwrap_or_else(|_| "./artifacts".into());
        let scratch = std::env::var("CITRATE_SCRATCH").unwrap_or_else(|_| "./scratch".into());
        tracing::info!(%store, %scratch, "training backend ready (nat)");
        Self {
            inner: citrate_training_worker::job_runner::NatJobRunner::new(store, scratch, worker),
        }
    }

    #[cfg(not(feature = "nat"))]
    fn new(_worker: ethereum_types::H160) -> Self {
        tracing::warn!(
            "built WITHOUT the `nat` feature: no training backend is compiled in, so \
             training jobs will be declined and reassigned. Rebuild with \
             `--features nat` (or `nat-cuda`) to execute them."
        );
        Self {}
    }

    #[cfg(feature = "nat")]
    async fn run(&self, job: JobSpec) -> anyhow::Result<String> {
        let result = self.inner.run(&job).await?;
        tracing::info!(
            job = %result.job,
            backend = result.backend,
            steps = result.steps.len(),
            epoch_root = ?result.epoch_root,
            seconds = result.seconds,
            "trained"
        );
        // A zone that never moved is the ADR-0012 failure, and it is far cheaper
        // to see it in a log line now than in a checkpoint months later.
        for (bucket, l2) in &result.zone_l2 {
            if *l2 == 0.0 {
                tracing::warn!(job = %result.job, %bucket, "bucket received NO gradient this job");
            }
        }
        Ok(serde_json::to_string(&result)?)
    }

    #[cfg(not(feature = "nat"))]
    async fn run(&self, job: JobSpec) -> anyhow::Result<String> {
        anyhow::bail!(
            "cannot execute job {} ({:?}): this worker was built without the `nat` \
             training backend. Declining so the lease expires and the coordinator \
             reassigns it.",
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
    let worker_id = wallet.address();
    let client = CoordinatorClient::new(&url, wallet);
    tracing::info!(worker = ?worker_id, coordinator = %url, "starting");

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

    let executor = Arc::new(Executor::new(worker_id));
    let check = running.clone();
    client
        .poll_loop(
            move |job| {
                let e = executor.clone();
                async move { e.run(job).await }
            },
            move || check.load(Ordering::SeqCst),
        )
        .await?;

    tracing::info!("stopped");
    Ok(())
}
