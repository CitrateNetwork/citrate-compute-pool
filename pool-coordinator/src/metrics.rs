//! Prometheus metrics surface for `citrate-pool-coordinator`.
//!
//! Operators point Prometheus at the configured metrics address
//! (see `CITRATE_POOL_METRICS_ADDR`, e.g. `127.0.0.1:9091`) and
//! scrape `/metrics` in the standard text-exposition format.
//!
//! # Metrics catalogue
//!
//! | Metric | Type | Labels | Meaning |
//! |--------|------|--------|---------|
//! | `pool_coord_events_observed_total` | counter | — | ComputeRequested events decoded by the poller |
//! | `pool_coord_events_dispatched_total` | counter | `outcome={success,not_coord,error}` | handle_event terminal states |
//! | `pool_coord_poll_failures_total` | counter | — | tick-level poll failures (RPC errors etc.) |
//! | `pool_coord_rpc_latency_seconds` | histogram | `method` | per-RPC-method latency |
//! | `pool_coord_seen_events_cardinality` | gauge | — | size of dedup set |
//!
//! Adding a new metric: update the table above FIRST so operators
//! have a single authoritative catalogue, then add the recorder
//! call. Metric names are string literals per the `metrics` crate
//! macro requirements.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use metrics::{describe_counter, describe_gauge, describe_histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use once_cell::sync::OnceCell;

/// Metric name constants — documentation / grep anchors only. The
/// `metrics::counter!` macros require string literals at the call
/// site; DO NOT reference these from the macros. If you rename one,
/// grep the crate for the literal.
pub const EVENTS_OBSERVED: &str = "pool_coord_events_observed_total";
pub const EVENTS_DISPATCHED: &str = "pool_coord_events_dispatched_total";
pub const POLL_FAILURES: &str = "pool_coord_poll_failures_total";
pub const RPC_LATENCY: &str = "pool_coord_rpc_latency_seconds";
pub const SEEN_EVENTS_CARDINALITY: &str = "pool_coord_seen_events_cardinality";

static HANDLE: OnceCell<PrometheusHandle> = OnceCell::new();

/// Install the Prometheus recorder. Safe to call repeatedly — first
/// install wins. Returns true on first install, false on subsequent.
pub fn install_recorder() -> bool {
    HANDLE
        .get_or_try_init(|| {
            let recorder = PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            // Best-effort global install. Tests or re-init calls that
            // collide are tolerated; local handle stays valid.
            let _ = metrics::set_global_recorder(recorder);
            describe_metrics();
            Ok::<_, std::convert::Infallible>(handle)
        })
        .is_ok()
}

fn describe_metrics() {
    describe_counter!(
        "pool_coord_events_observed_total",
        "ComputeRequested events decoded by the poller (before dedup)"
    );
    describe_counter!(
        "pool_coord_events_dispatched_total",
        "handle_event terminal states, labelled by outcome={success,not_coord,error}"
    );
    describe_counter!(
        "pool_coord_poll_failures_total",
        "Tick-level poll failures (RPC errors, decode failures)"
    );
    describe_histogram!(
        "pool_coord_rpc_latency_seconds",
        "Per-RPC-method latency, labelled by method"
    );
    describe_gauge!(
        "pool_coord_seen_events_cardinality",
        "Current size of the reorg-dedup set"
    );
}

/// Axum handler that renders the current Prometheus snapshot.
async fn metrics_handler() -> Response {
    match HANDLE.get() {
        Some(h) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            h.render(),
        )
            .into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            "recorder not installed",
        )
            .into_response(),
    }
}

/// Spawn the metrics HTTP server on the given bind address. Returns
/// immediately; the server runs in a background tokio task.
///
/// Idempotent w.r.t. recorder installation — safe to call even if
/// another component installed first.
pub async fn spawn_metrics_server(bind: &str) -> std::io::Result<()> {
    install_recorder();
    let app = Router::new().route("/metrics", get(metrics_handler));
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(bind = bind, "metrics server listening on /metrics");
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "metrics server crashed");
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn install_and_render_produces_prometheus_output() {
        // First install should succeed.
        install_recorder();

        // Increment a counter and verify it appears in rendered output.
        metrics::counter!(EVENTS_OBSERVED).increment(7);
        metrics::counter!(
            EVENTS_DISPATCHED,
            "outcome" => "success"
        )
        .increment(3);

        let handle = HANDLE.get().expect("handle installed");
        let rendered = handle.render();

        assert!(
            rendered.contains("pool_coord_events_observed_total"),
            "render missing events_observed: {}",
            rendered
        );
        assert!(
            rendered.contains("pool_coord_events_dispatched_total"),
            "render missing events_dispatched: {}",
            rendered
        );
    }

    #[tokio::test]
    async fn second_install_is_noop() {
        // First install may or may not have run depending on test order.
        let _ = install_recorder();
        // Second install must not panic.
        let _ = install_recorder();
        // Counter increments still work.
        metrics::counter!(POLL_FAILURES).increment(1);
    }
}
