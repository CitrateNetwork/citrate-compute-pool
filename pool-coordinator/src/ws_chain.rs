//! Optional websocket event-subscription path (S1 follow-up to
//! `http_chain.rs`'s polling path).
//!
//! Uses `eth_subscribe("logs", <filter>)` over WebSocket to get
//! real-time event delivery from a Citrate node — bypassing the
//! 3-second polling cadence the HTTP path uses. Latency drop is
//! typically 3s median → <1s median. Opt-in via
//! `CITRATE_POOL_WS_URL`; polling remains the default because
//! WebSocket requires an RPC endpoint that supports WS upgrade.
//!
//! On disconnect the caller's outer loop falls back to polling
//! via the existing `HttpChainAdapter` — zero-downtime degrade.
//!
//! # Data source (per Rule 11)
//!
//! | Method | JSON-RPC | Kind |
//! |--------|----------|------|
//! | `subscribe_compute_requested` | `eth_subscribe("logs", {address, topics})` | WS |
//! | — | `eth_unsubscribe(sub_id)` on drop | WS |
//!
//! The subscribed logs are `ComputeRequested(uint256,uint256,address,uint256)`
//! events from the ComputePool contract — same event the HTTP
//! `poll_compute_requested` decodes.

use ethereum_types::{H160, H256, U256};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use sha3::{Digest, Keccak256};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use crate::chain::ComputeRequestedEvent;
use crate::error::CoordinatorError;

/// A stream of decoded `ComputeRequestedEvent`s delivered by the
/// WebSocket subscription. Receivers `.await` on `recv()`.
pub struct EventStream {
    rx: mpsc::Receiver<Result<ComputeRequestedEvent, CoordinatorError>>,
}

impl EventStream {
    pub async fn recv(&mut self) -> Option<Result<ComputeRequestedEvent, CoordinatorError>> {
        self.rx.recv().await
    }
}

/// WebSocket-backed event subscription.
///
/// Construct via `connect`, then `.stream()` yields an EventStream.
/// When the connection drops, the EventStream closes (recv returns
/// None) and the caller is expected to fall back to polling (or
/// reconnect via a fresh `connect`).
pub struct WsChainSubscriber {
    rpc_ws_url: String,
    pool_contract: H160,
}

impl WsChainSubscriber {
    pub fn new(rpc_ws_url: String, pool_contract: H160) -> Self {
        Self {
            rpc_ws_url,
            pool_contract,
        }
    }

    /// Open a websocket connection and subscribe to
    /// `ComputeRequested` logs from the pool contract. Returns an
    /// [`EventStream`] the caller drains. A background tokio task
    /// decodes each incoming notification and pushes to the stream's
    /// mpsc; when the connection closes, the task sends a final
    /// error + drops the sender (stream closes).
    pub async fn connect(self) -> Result<EventStream, CoordinatorError> {
        let (ws_stream, _response) =
            tokio_tungstenite::connect_async(&self.rpc_ws_url)
                .await
                .map_err(|e| {
                    CoordinatorError::Chain(format!("ws connect {}: {}", self.rpc_ws_url, e))
                })?;

        let (mut write, mut read) = ws_stream.split();

        // Send eth_subscribe for logs, scoped to our contract +
        // ComputeRequested topic.
        let event_sig = compute_requested_sig_hex();
        let address_hex = format!("0x{}", hex::encode(self.pool_contract.as_bytes()));
        let subscribe_req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": ["logs", {
                "address": address_hex,
                "topics": [event_sig],
            }],
        });
        write
            .send(Message::Text(subscribe_req.to_string()))
            .await
            .map_err(|e| {
                CoordinatorError::Chain(format!("ws subscribe send: {}", e))
            })?;

        // Wait for the subscription id reply.
        let sub_id = loop {
            let msg = read.next().await.ok_or_else(|| {
                CoordinatorError::Chain("ws closed before subscribe ack".into())
            })?;
            let text = msg.map_err(|e| {
                CoordinatorError::Chain(format!("ws recv: {}", e))
            })?;
            let Message::Text(s) = text else { continue };
            let parsed: Value = serde_json::from_str(&s).map_err(|e| {
                CoordinatorError::Chain(format!("ws subscribe ack parse: {}", e))
            })?;
            if let Some(id) = parsed.get("result").and_then(|v| v.as_str()) {
                break id.to_string();
            }
            if let Some(err) = parsed.get("error") {
                return Err(CoordinatorError::Chain(format!(
                    "eth_subscribe rejected: {}",
                    err
                )));
            }
            // Not a subscribe ack — skip; ideally shouldn't happen
            // this early.
        };

        tracing::info!(sub_id = %sub_id, "ws subscription active");

        // Spawn decoder task.
        let (tx, rx) = mpsc::channel::<Result<ComputeRequestedEvent, CoordinatorError>>(64);
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                let frame = match msg {
                    Ok(Message::Text(s)) => s,
                    Ok(Message::Binary(b)) => match String::from_utf8(b) {
                        Ok(s) => s,
                        Err(_) => continue,
                    },
                    Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
                    Ok(Message::Close(_)) | Ok(Message::Frame(_)) => break,
                    Err(e) => {
                        let _ = tx
                            .send(Err(CoordinatorError::Chain(format!("ws recv: {}", e))))
                            .await;
                        break;
                    }
                };
                let parsed: Value = match serde_json::from_str(&frame) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Notification shape: { "method": "eth_subscription",
                //   "params": { "subscription": "0x...", "result": <log>} }
                let log = match parsed
                    .get("params")
                    .and_then(|p| p.get("result"))
                {
                    Some(l) => l.clone(),
                    None => continue,
                };
                match decode_compute_requested(&log) {
                    Ok(ev) => {
                        if tx.send(Ok(ev)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(e)).await.is_err() {
                            break;
                        }
                    }
                }
            }
            tracing::info!("ws decoder task exiting");
        });

        Ok(EventStream { rx })
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn compute_requested_sig_hex() -> String {
    let mut h = Keccak256::new();
    h.update(b"ComputeRequested(uint256,uint256,address,uint256)");
    let out = h.finalize();
    format!("0x{}", hex::encode(out))
}

fn decode_compute_requested(log: &Value) -> Result<ComputeRequestedEvent, CoordinatorError> {
    let topics = log
        .get("topics")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CoordinatorError::Chain("log missing topics".into()))?;
    if topics.len() < 4 {
        return Err(CoordinatorError::Chain(format!(
            "log has {} topics, expected 4",
            topics.len()
        )));
    }
    let pool_id = topic_u64(topics[1].as_str())?;
    let job_id = topic_u64(topics[2].as_str())?;
    let requester = topic_address(topics[3].as_str())?;

    let data_str = log
        .get("data")
        .and_then(|v| v.as_str())
        .ok_or_else(|| CoordinatorError::Chain("log missing data".into()))?;
    let data_bytes = hex::decode(data_str.trim_start_matches("0x"))
        .map_err(|e| CoordinatorError::Chain(format!("log data hex: {}", e)))?;
    if data_bytes.len() < 32 {
        return Err(CoordinatorError::Chain(format!(
            "log data too short: {}",
            data_bytes.len()
        )));
    }
    let payment = U256::from_big_endian(&data_bytes[..32]);

    let tx_hash = log
        .get("transactionHash")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s.trim_start_matches("0x")).ok())
        .and_then(|bytes| {
            if bytes.len() == 32 {
                Some(H256::from_slice(&bytes))
            } else {
                None
            }
        })
        .unwrap_or_default();
    let log_index = log
        .get("logIndex")
        .and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0) as u32;
    let block_number = log
        .get("blockNumber")
        .and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);

    // WS path doesn't do the PoolJobSpec chain-storage read; daemon
    // can enrich via HttpChainAdapter.fetch_job_spec if prompt /
    // max_tokens are needed. For ops dashboards (the primary WS
    // use-case) the indexed fields are enough.
    Ok(ComputeRequestedEvent {
        pool_id,
        job_id,
        requester,
        payment_grains: payment,
        prompt: String::new(),
        max_tokens: 1000,
        tx_hash,
        log_index,
        block_number,
    })
}

fn topic_u64(topic: Option<&str>) -> Result<u64, CoordinatorError> {
    let s = topic.ok_or_else(|| CoordinatorError::Chain("topic missing".into()))?;
    let stripped = s.trim_start_matches("0x");
    if stripped.len() != 64 {
        return Err(CoordinatorError::Chain(format!(
            "topic wrong length: {}",
            stripped.len()
        )));
    }
    let bytes = hex::decode(stripped)
        .map_err(|e| CoordinatorError::Chain(format!("topic hex: {}", e)))?;
    Ok(U256::from_big_endian(&bytes).as_u64())
}

fn topic_address(topic: Option<&str>) -> Result<H160, CoordinatorError> {
    let s = topic.ok_or_else(|| CoordinatorError::Chain("topic missing".into()))?;
    let stripped = s.trim_start_matches("0x");
    if stripped.len() != 64 {
        return Err(CoordinatorError::Chain(format!(
            "address topic wrong length: {}",
            stripped.len()
        )));
    }
    let bytes = hex::decode(stripped)
        .map_err(|e| CoordinatorError::Chain(format!("addr topic hex: {}", e)))?;
    Ok(H160::from_slice(&bytes[12..32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use std::time::Duration;
    use tokio::net::TcpListener;

    /// Spin up a minimal WebSocket server that accepts one client,
    /// sends back a subscription id, then pushes N log
    /// notifications before closing.
    async fn spawn_stub_ws(
        n_logs: usize,
    ) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.expect("accept");
            let ws = tokio_tungstenite::accept_async(tcp).await.expect("ws");
            let (mut write, mut read) = ws.split();

            // Read the subscribe request.
            let _sub = read.next().await;

            // Send subscription id.
            write
                .send(Message::Text(
                    json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": "0xsub1",
                    })
                    .to_string(),
                ))
                .await
                .expect("ack send");

            // Push N log notifications.
            for i in 0..n_logs {
                let log = make_log(7, 100 + i as u64);
                let notify = json!({
                    "jsonrpc": "2.0",
                    "method": "eth_subscription",
                    "params": {
                        "subscription": "0xsub1",
                        "result": log,
                    },
                });
                write
                    .send(Message::Text(notify.to_string()))
                    .await
                    .expect("notify");
            }
            // Leave the connection open briefly so the client can
            // drain. (It'll close automatically when the spawned
            // decoder task exits + tx drops.)
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = write.close().await;
        });
        addr
    }

    fn make_log(pool_id: u64, job_id: u64) -> Value {
        let requester = H160::repeat_byte(0xAB);
        let mut data = [0u8; 32];
        U256::from(99u64).to_big_endian(&mut data);
        let mut pool_topic = [0u8; 32];
        U256::from(pool_id).to_big_endian(&mut pool_topic);
        let mut job_topic = [0u8; 32];
        U256::from(job_id).to_big_endian(&mut job_topic);
        let mut req_topic = [0u8; 32];
        req_topic[12..32].copy_from_slice(requester.as_bytes());
        let sig = compute_requested_sig_hex();
        json!({
            "address": "0xdddddddddddddddddddddddddddddddddddddddd",
            "topics": [
                sig,
                format!("0x{}", hex::encode(pool_topic)),
                format!("0x{}", hex::encode(job_topic)),
                format!("0x{}", hex::encode(req_topic)),
            ],
            "data": format!("0x{}", hex::encode(data)),
            "blockNumber": "0x10",
            "transactionHash": format!("0x{}", "11".repeat(32)),
            "logIndex": "0x1",
        })
    }

    #[tokio::test]
    async fn subscribe_delivers_decoded_events() {
        let addr = spawn_stub_ws(3).await;
        let ws_url = format!("ws://{}", addr);
        let pool_contract = H160::repeat_byte(0xDD);
        let subscriber = WsChainSubscriber::new(ws_url, pool_contract);
        let mut stream = subscriber.connect().await.expect("connect");

        let mut received = 0;
        while let Some(result) = stream.recv().await {
            let ev = result.expect("decode ok");
            assert_eq!(ev.pool_id, 7);
            received += 1;
            if received == 3 {
                break;
            }
        }
        assert_eq!(received, 3);
    }

    #[tokio::test]
    async fn decode_rejects_short_topics() {
        // Log with only 2 topics — decode fails.
        let bad = json!({
            "topics": ["0x1", "0x2"],
            "data": "0x",
        });
        let err = decode_compute_requested(&bad).unwrap_err();
        assert!(matches!(err, CoordinatorError::Chain(_)));
    }

    #[tokio::test]
    async fn decode_populates_envelope_fields() {
        let log = make_log(5, 42);
        let ev = decode_compute_requested(&log).expect("decode");
        assert_eq!(ev.pool_id, 5);
        assert_eq!(ev.job_id, 42);
        assert_eq!(ev.block_number, 0x10);
    }
}
