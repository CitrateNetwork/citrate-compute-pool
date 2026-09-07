//! HTTP-backed [`PipelineChainClient`] against `ComputePoolPipeline` (S1).
//!
//! Reads go through `eth_call`; writes go through
//! `eth_sendRawTransaction` followed by an `eth_getTransactionReceipt`
//! poll (exponential backoff up to ~30s).
//!
//! Mirrors the pattern in `pool-coordinator/src/http_chain.rs` and
//! `http_chain_training.rs` — the three files should be kept
//! structurally similar until the shared helpers are factored out
//! post-pilot.
//!
//! # Data sources (per Rule 11)
//!
//! | Method | Contract.method | Kind |
//! |--------|-----------------|------|
//! | `request_snapshot` | `ComputePoolPipeline.requests(uint256)` (auto-generated mapping getter) | view |
//! | `stage_owner` | `ComputePoolPipeline.getStageOwner(uint256,uint32)` | view |
//! | `advance_request` | `ComputePoolPipeline.advanceRequest(uint256)` | write |
//!
//! # What is NOT in this slice
//!
//! - Payment accounting reads (`paymentEarned(jobId,worker)`): workers
//!   learn earnings via events in the real daemon loop.
//! - Fault/reassign writes (`faultStage`, `reassignStage`): separate
//!   chain-ops path, not on the happy advance-request flow.
//! - Attestation checks: performed on-chain in `advanceRequest`; the
//!   client just forwards the write.

use std::time::Duration;

use async_trait::async_trait;
use ethereum_types::{H160, H256, U256};
use serde_json::{json, Value};
use sha3::{Digest, Keccak256};

use crate::pipeline::{
    PipelineChainClient, PipelineChainError, PipelineJobId, PipelineRequestId,
    PipelineRequestSnapshot, PipelineRequestState, StageIndex,
};
use crate::types::WorkerAddress;
use crate::wallet::{tx_hash_of_signed, Eip1559Tx, Wallet};

/// Gas limit for `advanceRequest`. The contract body is small —
/// reads the Request + Job, one attestation check, two storage
/// updates, one event emit. 500k is generous.
const WRITE_GAS_LIMIT: u64 = 500_000;

const DEFAULT_PRIORITY_FEE_WEI: u64 = 1_000_000_000;
const RECEIPT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const RECEIPT_POLL_INITIAL: Duration = Duration::from_millis(250);
const RECEIPT_POLL_MAX: Duration = Duration::from_millis(500);
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
struct Selectors {
    requests: [u8; 4],
    get_stage_owner: [u8; 4],
    advance_request: [u8; 4],
}

impl Selectors {
    fn compute() -> Self {
        Self {
            // Public mapping getter: `requests(uint256)` — auto-
            // generated from `mapping(uint256 => Request) public requests`.
            requests: selector("requests(uint256)"),
            get_stage_owner: selector("getStageOwner(uint256,uint32)"),
            advance_request: selector("advanceRequest(uint256)"),
        }
    }
}

/// HTTP-backed [`PipelineChainClient`] against a live Citrate node.
#[derive(Clone)]
pub struct HttpPipelineChainClient {
    rpc_url: String,
    chain_id: u64,
    contract_addr: H160,
    wallet: Wallet,
    http: reqwest::Client,
    selectors: Selectors,
}

impl HttpPipelineChainClient {
    /// Construct a client pointed at `rpc_url`, signing writes to
    /// `contract_addr` with `wallet` on chain `chain_id`.
    pub fn new(
        rpc_url: String,
        chain_id: u64,
        contract_addr: H160,
        wallet: Wallet,
    ) -> Self {
        // CP-B-006 / CP-B-012: redirect-safe, timeout-bounded, fail-closed.
        let http = crate::outbound::redirect_safe_client(HTTP_TIMEOUT);
        Self {
            rpc_url,
            chain_id,
            contract_addr,
            wallet,
            http,
            selectors: Selectors::compute(),
        }
    }

    async fn rpc(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Value, PipelineChainError> {
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        });
        let resp = self
            .http
            .post(&self.rpc_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                PipelineChainError::WrongState(format!("{} transport: {}", method, e))
            })?;
        let value: Value = resp.json().await.map_err(|e| {
            PipelineChainError::WrongState(format!("{} decode: {}", method, e))
        })?;
        if let Some(err) = value.get("error").filter(|v| !v.is_null()) {
            return Err(PipelineChainError::WrongState(format!(
                "{} rpc error: {}",
                method, err
            )));
        }
        value.get("result").cloned().ok_or_else(|| {
            PipelineChainError::WrongState(format!("{} missing result", method))
        })
    }

    async fn eth_call(&self, data: &[u8]) -> Result<Vec<u8>, PipelineChainError> {
        let to = format!("0x{}", hex::encode(self.contract_addr.as_bytes()));
        let data_hex = format!("0x{}", hex::encode(data));
        let params = json!([
            { "to": to, "data": data_hex },
            "latest",
        ]);
        let result = self.rpc("eth_call", params).await?;
        let hex_str = result.as_str().ok_or_else(|| {
            PipelineChainError::WrongState("eth_call result not a string".into())
        })?;
        hex::decode(hex_str.trim_start_matches("0x")).map_err(|e| {
            PipelineChainError::WrongState(format!("eth_call bad hex: {}", e))
        })
    }

    async fn fetch_nonce(&self) -> Result<u64, PipelineChainError> {
        let addr = format!("0x{}", hex::encode(self.wallet.address().as_bytes()));
        let result = self
            .rpc("eth_getTransactionCount", json!([addr, "pending"]))
            .await?;
        let hex_str = result.as_str().ok_or_else(|| {
            PipelineChainError::WrongState(
                "eth_getTransactionCount not a string".into(),
            )
        })?;
        parse_hex_u64(hex_str).map_err(|e| {
            PipelineChainError::WrongState(format!("nonce decode: {}", e))
        })
    }

    async fn fetch_gas_price(&self) -> Result<U256, PipelineChainError> {
        let result = self.rpc("eth_gasPrice", json!([])).await?;
        let hex_str = result.as_str().ok_or_else(|| {
            PipelineChainError::WrongState("eth_gasPrice not a string".into())
        })?;
        parse_hex_u256(hex_str).map_err(|e| {
            PipelineChainError::WrongState(format!("gas price decode: {}", e))
        })
    }

    async fn send_write(
        &self,
        calldata: Vec<u8>,
    ) -> Result<H256, PipelineChainError> {
        let nonce = self.fetch_nonce().await?;
        let gas_price = self.fetch_gas_price().await?;
        let priority = U256::from(DEFAULT_PRIORITY_FEE_WEI);
        let max_fee = gas_price.saturating_mul(U256::from(2u64)).saturating_add(priority);

        let tx = Eip1559Tx {
            chain_id: self.chain_id,
            nonce,
            max_priority_fee_per_gas: priority,
            max_fee_per_gas: max_fee,
            gas_limit: WRITE_GAS_LIMIT,
            to: self.contract_addr,
            value: U256::zero(),
            data: calldata,
        };
        let signed = self
            .wallet
            .sign_eip1559(&tx)
            .map_err(|e| PipelineChainError::WrongState(format!("sign: {}", e)))?;
        let tx_hash = tx_hash_of_signed(&signed);
        let signed_hex = format!("0x{}", hex::encode(&signed));

        let _ = self
            .rpc("eth_sendRawTransaction", json!([signed_hex]))
            .await?;

        let _block_number = self.wait_for_receipt(tx_hash).await?;
        Ok(tx_hash)
    }

    async fn wait_for_receipt(
        &self,
        tx_hash: H256,
    ) -> Result<Option<u64>, PipelineChainError> {
        let hash_hex = format!("0x{}", hex::encode(tx_hash.as_bytes()));
        let deadline = tokio::time::Instant::now() + RECEIPT_WAIT_TIMEOUT;
        let mut delay = RECEIPT_POLL_INITIAL;

        while tokio::time::Instant::now() < deadline {
            let result = self
                .rpc("eth_getTransactionReceipt", json!([hash_hex]))
                .await?;
            if !result.is_null() {
                let status = result
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0x1");
                if parse_hex_u64(status).unwrap_or(1) == 0 {
                    return Err(PipelineChainError::WrongState(format!(
                        "tx {:?} reverted",
                        tx_hash
                    )));
                }
                let bn = result
                    .get("blockNumber")
                    .and_then(|v| v.as_str())
                    .and_then(|s| parse_hex_u64(s).ok());
                return Ok(bn);
            }
            tokio::time::sleep(delay).await;
            delay = std::cmp::min(delay.saturating_mul(2), RECEIPT_POLL_MAX);
        }
        tracing::warn!(tx_hash = ?tx_hash, "receipt not found within timeout");
        Ok(None)
    }

    /// Fetch the latest block number from the node for the event
    /// poller's `to_block` bound.
    pub async fn latest_block(&self) -> Result<u64, PipelineChainError> {
        let result = self.rpc("eth_blockNumber", serde_json::json!([])).await?;
        let hex_str = result.as_str().ok_or_else(|| {
            PipelineChainError::WrongState("eth_blockNumber not a string".into())
        })?;
        u64::from_str_radix(hex_str.trim_start_matches("0x"), 16).map_err(|e| {
            PipelineChainError::WrongState(format!("blockNumber decode: {}", e))
        })
    }

    /// Poll all ComputePoolPipeline events in `[from_block,
    /// to_block]` optionally filtered by indexed jobId. Returns
    /// raw logs the caller classifies via `crate::events::classify`.
    pub async fn poll_raw_logs(
        &self,
        from_block: u64,
        to_block: u64,
        job_id_filter: Option<u64>,
    ) -> Result<Vec<crate::events::RawLog>, PipelineChainError> {
        let address_hex = format!("0x{}", hex::encode(self.contract_addr.as_bytes()));
        let params = match job_id_filter {
            Some(id) => serde_json::json!([{
                "fromBlock": format!("0x{:x}", from_block),
                "toBlock":   format!("0x{:x}", to_block),
                "address":   address_hex,
                "topics":    crate::events::job_id_topic_filter(id),
            }]),
            None => serde_json::json!([{
                "fromBlock": format!("0x{:x}", from_block),
                "toBlock":   format!("0x{:x}", to_block),
                "address":   address_hex,
            }]),
        };
        let result = self.rpc("eth_getLogs", params).await?;
        let arr = result.as_array().ok_or_else(|| {
            PipelineChainError::WrongState("eth_getLogs not array".into())
        })?;
        let mut out = Vec::with_capacity(arr.len());
        for entry in arr {
            if let Some(log) = crate::events::decode_log_entry(entry) {
                out.push(log);
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl PipelineChainClient for HttpPipelineChainClient {
    async fn request_snapshot(
        &self,
        request_id: PipelineRequestId,
    ) -> Result<PipelineRequestSnapshot, PipelineChainError> {
        // ABI-encoded return for a public `mapping(uint => Request)`
        // getter. Request has 5 static-typed fields, so the return is
        // a flat 5×32 = 160 bytes (no head/tail indirection):
        //   [  0..  32] jobId (uint256)
        //   [ 32..  64] requester (address)
        //   [ 64..  96] progress (uint32)
        //   [ 96.. 128] state (uint8 enum)
        //   [128.. 160] escrow (uint128)
        const EXPECTED_LEN: usize = 160;
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.requests);
        data.extend_from_slice(&u256_word(U256::from(request_id)));
        let ret = self.eth_call(&data).await?;
        if ret.len() < EXPECTED_LEN {
            return Err(PipelineChainError::WrongState(format!(
                "requests return too short: {} bytes (expected {})",
                ret.len(),
                EXPECTED_LEN,
            )));
        }

        let job_id_u256 = U256::from_big_endian(&ret[0..32]);
        let requester = H160::from_slice(&ret[32 + 12..32 + 32]);
        // If requester is zero, treat as unknown request. A live
        // Solidity mapping getter returns all-zeros for absent keys.
        if requester == H160::zero() && job_id_u256.is_zero() {
            return Err(PipelineChainError::UnknownRequest(request_id));
        }
        let job_id = if job_id_u256 > U256::from(u64::MAX) {
            return Err(PipelineChainError::WrongState(format!(
                "jobId out of u64 range: {}",
                job_id_u256
            )));
        } else {
            job_id_u256.as_u64() as PipelineJobId
        };
        let progress = u32_from_word(&ret[64..96])?;
        let state_byte = ret[127]; // uint8 sits in last byte of word
        let state = request_state_from_byte(state_byte)?;

        Ok(PipelineRequestSnapshot {
            request_id,
            job_id,
            requester,
            progress,
            state,
        })
    }

    async fn stage_owner(
        &self,
        job_id: PipelineJobId,
        stage: StageIndex,
    ) -> Result<Option<WorkerAddress>, PipelineChainError> {
        let mut data = Vec::with_capacity(4 + 64);
        data.extend_from_slice(&self.selectors.get_stage_owner);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        data.extend_from_slice(&u256_word(U256::from(stage)));
        let ret = self.eth_call(&data).await?;
        if ret.len() < 32 {
            return Err(PipelineChainError::WrongState(format!(
                "getStageOwner return too short: {} bytes",
                ret.len()
            )));
        }
        let owner = H160::from_slice(&ret[12..32]);
        // Contract signals "unowned (faulted)" with address(0).
        if owner == H160::zero() {
            Ok(None)
        } else {
            Ok(Some(owner))
        }
    }

    async fn advance_request(
        &self,
        request_id: PipelineRequestId,
        _caller: WorkerAddress,
    ) -> Result<PipelineRequestState, PipelineChainError> {
        // `caller` is trait-shape-compat with the mock; on chain, the
        // caller is msg.sender from the signing wallet, which the
        // daemon loads from the operator's keystore.
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.advance_request);
        data.extend_from_slice(&u256_word(U256::from(request_id)));
        self.send_write(data).await?;

        // After the write lands, read back the latest state. This
        // matches the mock's return signature (which carries the
        // new state). If the post-write read fails, the tx did
        // still go through — we surface the read error.
        let snap = self.request_snapshot(request_id).await?;
        Ok(snap.state)
    }
}

// ── ABI helpers (private) ────────────────────────────────────────

fn selector(sig: &str) -> [u8; 4] {
    let mut h = Keccak256::new();
    h.update(sig.as_bytes());
    let out = h.finalize();
    [out[0], out[1], out[2], out[3]]
}

fn u256_word(v: U256) -> [u8; 32] {
    let mut buf = [0u8; 32];
    v.to_big_endian(&mut buf);
    buf
}

fn u32_from_word(word: &[u8]) -> Result<u32, PipelineChainError> {
    if word.len() != 32 {
        return Err(PipelineChainError::WrongState(format!(
            "u32 word length {} (expected 32)",
            word.len()
        )));
    }
    let v = U256::from_big_endian(word);
    if v > U256::from(u32::MAX) {
        Err(PipelineChainError::WrongState(format!(
            "u32 value out of range: {}",
            v
        )))
    } else {
        Ok(v.as_u32())
    }
}

fn request_state_from_byte(
    b: u8,
) -> Result<PipelineRequestState, PipelineChainError> {
    // Matches Solidity enum RequestState:
    //   0 = Created, 1 = InFlight, 2 = Completed, 3 = Failed.
    // Our Rust enum has no `Created` (the mock starts at InFlight on
    // submit); we coalesce 0→InFlight since the buyer-webapp-visible
    // lifecycle only surfaces post-submit states. If a future flow
    // exposes Created explicitly, add a variant to
    // PipelineRequestState.
    match b {
        0 | 1 => Ok(PipelineRequestState::InFlight),
        2 => Ok(PipelineRequestState::Completed),
        3 => Ok(PipelineRequestState::Failed),
        other => Err(PipelineChainError::WrongState(format!(
            "unknown RequestState discriminant: {}",
            other
        ))),
    }
}

fn parse_hex_u64(s: &str) -> Result<u64, String> {
    let trimmed = s.trim_start_matches("0x");
    if trimmed.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(trimmed, 16).map_err(|e| e.to_string())
}

fn parse_hex_u256(s: &str) -> Result<U256, String> {
    let trimmed = s.trim_start_matches("0x");
    if trimmed.is_empty() {
        return Ok(U256::zero());
    }
    U256::from_str_radix(trimmed, 16).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::{HashMap, VecDeque};
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::routing::post;
    use axum::Json;
    use ethereum_types::Address;
    use tokio::net::TcpListener;

    const TEST_HEX: &str =
        "0000000000000000000000000000000000000000000000000000000000000001";

    #[derive(Default, Clone)]
    struct RpcLog {
        calls: Arc<Mutex<Vec<(String, Value)>>>,
    }

    impl RpcLog {
        fn methods(&self) -> Vec<String> {
            self.calls
                .lock()
                .expect("rpc log poisoned")
                .iter()
                .map(|(m, _)| m.clone())
                .collect()
        }
    }

    #[derive(Clone)]
    struct StubState {
        log: RpcLog,
        responses: Arc<Mutex<HashMap<String, VecDeque<Value>>>>,
    }

    impl StubState {
        fn new() -> Self {
            Self {
                log: RpcLog::default(),
                responses: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        fn queue(&self, method: &str, response: Value) {
            self.responses
                .lock()
                .expect("responses poisoned")
                .entry(method.to_string())
                .or_default()
                .push_back(response);
        }

        fn take(&self, method: &str) -> Option<Value> {
            self.responses
                .lock()
                .expect("responses poisoned")
                .get_mut(method)
                .and_then(|q| q.pop_front())
        }

        fn queue_write_happy_path(&self, nonce_hex: &str) {
            self.queue("eth_getTransactionCount", json!(nonce_hex));
            self.queue("eth_gasPrice", json!("0x3b9aca00"));
            self.queue("eth_sendRawTransaction", json!("0xdeadbeef"));
            self.queue(
                "eth_getTransactionReceipt",
                json!({ "status": "0x1", "blockNumber": "0x10" }),
            );
        }
    }

    async fn rpc_handler(
        State(state): State<StubState>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let method = body
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let params = body.get("params").cloned().unwrap_or(Value::Null);
        state
            .log
            .calls
            .lock()
            .expect("rpc log poisoned")
            .push((method.clone(), params));
        let id = body.get("id").cloned().unwrap_or(json!(1));
        let result = state.take(&method).unwrap_or(Value::Null);
        Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }))
    }

    async fn spawn_stub_rpc(state: StubState) -> SocketAddr {
        let app = axum::Router::new()
            .route("/", post(rpc_handler))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        addr
    }

    fn contract_addr() -> H160 {
        H160::repeat_byte(0xDD)
    }

    fn make_client(rpc_url: String) -> HttpPipelineChainClient {
        let wallet = Wallet::from_hex(TEST_HEX).expect("wallet");
        HttpPipelineChainClient::new(rpc_url, 40204, contract_addr(), wallet)
    }

    /// Build the ABI-encoded return for the auto-generated
    /// `requests(uint256)` getter: 5 × 32 = 160 bytes.
    fn encode_request(
        job_id: u64,
        requester: H160,
        progress: u32,
        state: u8,
        escrow: u128,
    ) -> String {
        let mut out = vec![0u8; 160];
        U256::from(job_id).to_big_endian(&mut out[0..32]);
        out[32 + 12..32 + 32].copy_from_slice(requester.as_bytes());
        U256::from(progress).to_big_endian(&mut out[64..96]);
        out[127] = state; // uint8 in last byte of word
        U256::from(escrow).to_big_endian(&mut out[128..160]);
        format!("0x{}", hex::encode(out))
    }

    fn encode_address_word(addr: H160) -> String {
        let mut buf = vec![0u8; 32];
        buf[12..32].copy_from_slice(addr.as_bytes());
        format!("0x{}", hex::encode(buf))
    }

    // ── Tests ────────────────────────────────────────────────────

    #[test]
    fn selectors_are_distinct_and_four_bytes() {
        let s = Selectors::compute();
        let all = [s.requests, s.get_stage_owner, s.advance_request];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(
                    all[i], all[j],
                    "selector collision at {} vs {}",
                    i, j
                );
            }
        }
    }

    #[tokio::test]
    async fn advance_request_signs_writes_and_reads_back_state() {
        let state = StubState::new();
        // Write path.
        state.queue_write_happy_path("0x3");
        // Post-write snapshot read: Completed state (progress = 4
        // out of stage_count; in the contract completed is signaled
        // by state=2).
        state.queue(
            "eth_call",
            json!(encode_request(
                7,
                Address::repeat_byte(0xAA),
                4,
                2, // Completed
                0,
            )),
        );
        let addr = spawn_stub_rpc(state.clone()).await;

        let client = make_client(format!("http://{}", addr));
        let caller = Address::repeat_byte(0x77);
        let result = client
            .advance_request(42, caller)
            .await
            .expect("advance");

        assert_eq!(result, PipelineRequestState::Completed);

        // The write path ran: nonce → gas → send → receipt → then the
        // read (eth_call).
        let methods = state.log.methods();
        assert_eq!(methods[0], "eth_getTransactionCount");
        assert_eq!(methods[1], "eth_gasPrice");
        assert_eq!(methods[2], "eth_sendRawTransaction");
        assert_eq!(methods[3], "eth_getTransactionReceipt");
        assert_eq!(methods[4], "eth_call");
    }

    #[tokio::test]
    async fn stage_owner_returns_some_for_owned_and_none_for_zero() {
        let state = StubState::new();
        let owner = Address::repeat_byte(0x12);
        state.queue("eth_call", json!(encode_address_word(owner)));
        // Second call returns all-zeros address → Option::None.
        state.queue(
            "eth_call",
            json!(format!("0x{}", hex::encode(vec![0u8; 32]))),
        );
        let addr = spawn_stub_rpc(state).await;

        let client = make_client(format!("http://{}", addr));
        let got = client.stage_owner(9, 2).await.expect("call");
        assert_eq!(got, Some(owner));

        let got2 = client.stage_owner(9, 3).await.expect("call2");
        assert_eq!(got2, None);
    }

    #[tokio::test]
    async fn request_snapshot_decodes_request_fields() {
        let state = StubState::new();
        let requester = Address::repeat_byte(0xBC);
        state.queue(
            "eth_call",
            json!(encode_request(
                11,
                requester,
                2,
                1,                           // InFlight
                3_000_000_000_000_000_000u128, // escrow (3 ether)
            )),
        );
        let addr = spawn_stub_rpc(state).await;
        let client = make_client(format!("http://{}", addr));

        let snap = client.request_snapshot(42).await.expect("snapshot");
        assert_eq!(snap.request_id, 42);
        assert_eq!(snap.job_id, 11);
        assert_eq!(snap.requester, requester);
        assert_eq!(snap.progress, 2);
        assert_eq!(snap.state, PipelineRequestState::InFlight);
    }

    #[tokio::test]
    async fn request_snapshot_surfaces_unknown_for_zero_payload() {
        let state = StubState::new();
        // All-zero payload — Solidity mapping getter response when
        // the requestId doesn't exist.
        state.queue(
            "eth_call",
            json!(format!("0x{}", hex::encode(vec![0u8; 160]))),
        );
        let addr = spawn_stub_rpc(state).await;
        let client = make_client(format!("http://{}", addr));

        let err = client.request_snapshot(99).await.expect_err("unknown");
        assert!(matches!(err, PipelineChainError::UnknownRequest(_)));
    }

    #[tokio::test]
    async fn request_snapshot_rejects_short_return() {
        let state = StubState::new();
        state.queue("eth_call", json!("0xabcd"));
        let addr = spawn_stub_rpc(state).await;
        let client = make_client(format!("http://{}", addr));

        let err = client.request_snapshot(1).await.expect_err("short");
        assert!(matches!(err, PipelineChainError::WrongState(_)));
    }

    #[test]
    fn request_state_from_byte_maps_variants() {
        assert_eq!(
            request_state_from_byte(0).expect("0"),
            PipelineRequestState::InFlight
        );
        assert_eq!(
            request_state_from_byte(1).expect("1"),
            PipelineRequestState::InFlight
        );
        assert_eq!(
            request_state_from_byte(2).expect("2"),
            PipelineRequestState::Completed
        );
        assert_eq!(
            request_state_from_byte(3).expect("3"),
            PipelineRequestState::Failed
        );
        assert!(request_state_from_byte(4).is_err());
    }
}
