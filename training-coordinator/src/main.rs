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
//!
//! Admission policy (PBA-L3b-001):
//!
//! - `CITRATE_COORDINATOR_H01_WORKERS` — comma-separated worker addresses the
//!   operator has vetted for H-01 (ladder) work. A self-reported H-01 probe from
//!   any other key is granted `federated`. Unset means nobody is granted H-01.
//! - `CITRATE_COORDINATOR_OPEN_TIER` — highest tier granted to unvouched
//!   workers: `probe` (default) or `federated`. `CITRATE_COORDINATOR_H01_WORKERS`
//!   lists the vouched addresses, which may take any tier.
//! - `CITRATE_COORDINATOR_MAX_WORKERS_PER_SOURCE` — distinct unvouched keys one
//!   client address may register (default 16).
//! - `CITRATE_COORDINATOR_MAX_LEASES_PER_SOURCE` — live leases unvouched
//!   workers of one source group (IPv4 address, IPv6 /48) may hold (default 4).

use std::net::SocketAddr;
use std::sync::Arc;

use citrate_training_coordinator::api::{router, Coordinator};
use citrate_training_coordinator::job::JobSpec;
use citrate_training_coordinator::state::Policy;
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

    let policy = policy_from_env()?;
    if policy.trusted_h01.is_empty() {
        tracing::warn!(
            "CITRATE_COORDINATOR_H01_WORKERS is unset: no worker will be granted H-01 \
             (self-reported probes are not trusted for the ladder, PBA-L3b-001)"
        );
    }
    tracing::info!(
        h01_workers = policy.trusted_h01.len(),
        open_tier = ?policy.open_tier,
        max_workers_per_source = policy.max_workers_per_source,
        lease_window_secs = policy.lease_window_secs,
        max_leases_per_source = policy.max_leases_per_source,
        "admission policy"
    );
    let coord = Arc::new(Coordinator::open_with(Store::new(&state_path), policy)?);

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
    // Connect info feeds the per-source identity cap (PBA-L3b-001).
    axum::serve(
        listener,
        router(coord).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

fn policy_from_env() -> anyhow::Result<Policy> {
    let mut policy = Policy::default();
    if let Ok(raw) = std::env::var("CITRATE_COORDINATOR_H01_WORKERS") {
        policy.trusted_h01 = citrate_training_coordinator::state::parse_address_list(&raw)
            .map_err(|e| anyhow::anyhow!("CITRATE_COORDINATOR_H01_WORKERS: {e}"))?;
    }
    if let Ok(raw) = std::env::var("CITRATE_COORDINATOR_OPEN_TIER") {
        policy.open_tier = match raw.trim() {
            "probe" => citrate_training_coordinator::Capability::Probe,
            "federated" => citrate_training_coordinator::Capability::Federated,
            other => anyhow::bail!(
                "CITRATE_COORDINATOR_OPEN_TIER={other:?}: expected \"probe\" or \"federated\""
            ),
        };
    }
    if let Ok(raw) = std::env::var("CITRATE_COORDINATOR_MAX_WORKERS_PER_SOURCE") {
        let n: usize = raw.trim().parse().map_err(|e| {
            anyhow::anyhow!("CITRATE_COORDINATOR_MAX_WORKERS_PER_SOURCE={raw:?}: {e}")
        })?;
        anyhow::ensure!(
            n > 0,
            "CITRATE_COORDINATOR_MAX_WORKERS_PER_SOURCE must be > 0"
        );
        policy.max_workers_per_source = n;
    }
    if let Ok(raw) = std::env::var("CITRATE_COORDINATOR_MAX_LEASES_PER_SOURCE") {
        let n: usize = raw.trim().parse().map_err(|e| {
            anyhow::anyhow!("CITRATE_COORDINATOR_MAX_LEASES_PER_SOURCE={raw:?}: {e}")
        })?;
        anyhow::ensure!(
            n > 0,
            "CITRATE_COORDINATOR_MAX_LEASES_PER_SOURCE must be > 0"
        );
        policy.max_leases_per_source = n;
    }
    Ok(policy)
}
