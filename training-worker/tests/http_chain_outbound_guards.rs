//! The training-worker's JSON-RPC client, driven against local stub servers.
//!
//! Two properties of the signing leg, checked at the client's own entry points:
//!
//! 1. `verify_chain_id` refuses an RPC that reports a different chain id from
//!    the configured one, so the daemon never signs for the wrong network.
//! 2. A 3xx from the RPC is returned as an error, not followed: the request
//!    body is never re-sent to the `Location:` host.
//!
//! Every server binds 127.0.0.1:0. Nothing leaves the host.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::routing::post;
use citrate_training_worker::wallet::Wallet;
use citrate_training_worker::HttpChainClient;
use ethereum_types::H160;
use tokio::net::TcpListener;

// Anvil account #0, a well-known throwaway, NOT a real key.
const KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const CHAIN_ID: u64 = 40204;

async fn serve(app: axum::Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

/// A JSON-RPC stub that answers every call with `result`.
async fn spawn_rpc(result: &'static str) -> SocketAddr {
    serve(axum::Router::new().route(
        "/",
        post(move |_body: String| async move {
            axum::Json(serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": result}))
        }),
    ))
    .await
}

/// Counts every body POSTed to it, and answers like a correct RPC would.
async fn spawn_sink() -> (SocketAddr, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let app = axum::Router::new()
        .route(
            "/sink",
            post(
                |State(h): State<Arc<AtomicUsize>>, _body: String| async move {
                    h.fetch_add(1, Ordering::SeqCst);
                    axum::Json(serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": "0x9d0c"}))
                },
            ),
        )
        .with_state(hits.clone());
    (serve(app).await, hits)
}

/// Answers every POST with a 307 to the sink.
async fn spawn_redirector(sink: SocketAddr) -> SocketAddr {
    let location = format!("http://{sink}/sink");
    serve(axum::Router::new().route(
        "/",
        post(move |_body: String| {
            let location = location.clone();
            async move {
                (
                    axum::http::StatusCode::TEMPORARY_REDIRECT,
                    [(axum::http::header::LOCATION, location)],
                )
            }
        }),
    ))
    .await
}

fn client(url: String) -> HttpChainClient {
    HttpChainClient::new(
        url,
        CHAIN_ID,
        H160::repeat_byte(0xCC),
        Wallet::from_hex(KEY).expect("test key"),
    )
}

#[tokio::test]
async fn verify_chain_id_refuses_an_rpc_on_another_chain() {
    let addr = spawn_rpc("0x1").await;
    let out = client(format!("http://{addr}/")).verify_chain_id().await;
    let err = out.expect_err("an RPC reporting chain 1 must be refused for 40204");
    assert!(
        err.to_string().contains("chain_id mismatch"),
        "the refusal must name the chain id mismatch, got: {err}"
    );
}

#[tokio::test]
async fn verify_chain_id_accepts_the_configured_chain() {
    let addr = spawn_rpc("0x9d0c").await; // 40204
    client(format!("http://{addr}/"))
        .verify_chain_id()
        .await
        .expect("the configured chain id must verify");
}

#[tokio::test]
async fn a_redirecting_rpc_is_not_followed() {
    let (sink, hits) = spawn_sink().await;
    let redirector = spawn_redirector(sink).await;
    let out = client(format!("http://{redirector}/"))
        .verify_chain_id()
        .await;
    assert!(
        out.is_err(),
        "a 3xx must surface as an error, not as the redirect target's answer"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "the request body must not be re-sent to the redirect target"
    );
}
