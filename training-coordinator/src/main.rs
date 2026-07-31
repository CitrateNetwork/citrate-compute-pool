//! The coordinator daemon.
//!
//! ```sh
//! CITRATE_COORDINATOR_STATE=/var/lib/citrate/coordinator.json \
//! CITRATE_COORDINATOR_BIND=0.0.0.0:8088 \
//!   citrate-training-coordinator
//! ```
//!
//! Jobs are loaded from a catalogue file at boot (`CITRATE_COORDINATOR_JOBS`), a
//! JSON array of `JobSpec`. Adding work is editing that file and restarting —
//! deliberately, for now: a fleet of five machines does not need a job-authoring
//! API, and not having one means there is no privileged write path to secure.
//! Jobs already recorded keep their state; only unseen ids are added.

use std::sync::Arc;

use citrate_training_coordinator::api::{router, Coordinator};
use citrate_training_coordinator::job::JobSpec;
use citrate_training_coordinator::store::Store;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let state_path = std::env::var("CITRATE_COORDINATOR_STATE")
        .unwrap_or_else(|_| "./coordinator-state.json".into());
    let bind =
        std::env::var("CITRATE_COORDINATOR_BIND").unwrap_or_else(|_| "127.0.0.1:8088".into());

    let coord = Arc::new(Coordinator::open(Store::new(&state_path))?);

    if let Ok(jobs_path) = std::env::var("CITRATE_COORDINATOR_JOBS") {
        let raw = std::fs::read_to_string(&jobs_path)?;
        let specs: Vec<JobSpec> = serde_json::from_str(&raw)?;
        let known = coord.snapshot();
        let mut added = 0usize;
        for s in specs {
            // Never clobber a job that is mid-flight or finished.
            if known.jobs.contains_key(&s.id) {
                continue;
            }
            coord.add_job(s)?;
            added += 1;
        }
        tracing::info!(catalogue = %jobs_path, added, "job catalogue loaded");
    }

    let counts = coord.snapshot().counts();
    tracing::info!(
        pending = counts.pending,
        leased = counts.leased,
        done = counts.done,
        quarantined = counts.quarantined,
        workers = counts.workers,
        settlement = "shadow",
        "coordinator starting"
    );

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(%bind, "listening");
    axum::serve(listener, router(coord)).await?;
    Ok(())
}
