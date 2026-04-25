//! HTTP-backed `ChainAdapter` implementation (CM-05 slice-2).
//!
//! Real JSON-RPC client that speaks to a Citrate node via `reqwest`
//! and signs on-chain writes with a [`Wallet`]. Reads go through
//! `eth_call`; writes go through `eth_sendRawTransaction` followed by
//! an `eth_getTransactionReceipt` poll (exponential backoff up to
//! ~30s).
//!
//! ABI encoding is hand-rolled — the five selectors this adapter
//! needs all use fixed-size arguments (`uint256`, `address`) plus
//! a single dynamic `address[]` return. That's well within reach of
//! a dozen lines of helpers, and sidesteps pulling in `ethabi` as a
//! new workspace dependency.
//!
//! The selectors are computed at construction time (via `Keccak256`)
//! and cached on the struct — no per-call hashing overhead.
//!
//! # Data sources (per Rule 11)
//!
//! Every method names the on-chain contract + method it reads from
//! or writes to so the daemon never silently returns fake data:
//!
//! | Method | Contract.method | Kind |
//! |--------|-----------------|------|
//! | `coordinator_for` | `ComputePool.coordinatorFor(uint256,uint256)` | view |
//! | `pool_members` | `ComputePool.getPoolMembers(uint256)` + `ComputePool.getMember(uint256,address)` | view |
//! | `record_dispatch` | `ComputePool.recordDispatch(uint256)` | write |
//! | `complete_job` | `ComputePool.completeJob(uint256)` | write |
//! | `fail_job` | `ComputePool.failJob(uint256)` | write |

use std::time::Duration;

use async_trait::async_trait;
use ethereum_types::{H160, H256, U256};
use serde_json::{json, Value};
use sha3::{Digest, Keccak256};

use crate::chain::{ChainAdapter, ComputeRequestedEvent, PoolMemberInfo, RecordDispatchOutcome};
use crate::error::CoordinatorError;
use crate::wallet::{tx_hash_of_signed, Eip1559Tx, Wallet};

/// Conservative gas limit for the three `ComputePool` writes this
/// adapter submits (`recordDispatch`, `completeJob`, `failJob`). Each
/// touches a single storage slot plus emits one event — well under
/// 100k gas in practice. 200k gives ~2× headroom for any future
/// contract-side changes without needing a round-trip to
/// `eth_estimateGas` on every submit.
const WRITE_GAS_LIMIT: u64 = 200_000;

/// Minimum tip paid to the proposer (1 gwei). The daemon isn't
/// competing for inclusion priority, but a tip of zero risks being
/// dropped by some mempool policies.
const DEFAULT_PRIORITY_FEE_WEI: u64 = 1_000_000_000;

/// Max time we're willing to wait for a submitted tx to appear in a
/// block. After this we return a `Chain` error; the daemon loop
/// surfaces it to the operator. 30s comfortably covers the 1-2s
/// block target with room for a re-org or a slow peer.
const RECEIPT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Initial poll delay for `eth_getTransactionReceipt`. We double on
/// each miss until capped at [`RECEIPT_POLL_MAX`].
const RECEIPT_POLL_INITIAL: Duration = Duration::from_millis(250);

/// Upper bound on poll delay — past this we query every half second
/// until [`RECEIPT_WAIT_TIMEOUT`] expires.
const RECEIPT_POLL_MAX: Duration = Duration::from_millis(500);

/// Default per-request HTTP timeout for the JSON-RPC calls.
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

/// Precomputed 4-byte function selectors. Cheaper than recomputing
/// keccak on every call.
#[derive(Clone)]
struct Selectors {
    coordinator_for: [u8; 4],
    get_pool_members: [u8; 4],
    get_member: [u8; 4],
    get_job: [u8; 4],
    record_dispatch: [u8; 4],
    complete_job: [u8; 4],
    fail_job: [u8; 4],
}

impl Selectors {
    fn compute() -> Self {
        Self {
            coordinator_for: selector("coordinatorFor(uint256,uint256)"),
            get_pool_members: selector("getPoolMembers(uint256)"),
            get_member: selector("getMember(uint256,address)"),
            get_job: selector("getJob(uint256)"),
            record_dispatch: selector("recordDispatch(uint256)"),
            complete_job: selector("completeJob(uint256)"),
            fail_job: selector("failJob(uint256)"),
        }
    }
}

/// Default `max_tokens` when PoolJobSpec decode fails or returns
/// a zero value. Keeps provider calls from becoming no-ops.
const DEFAULT_MAX_TOKENS: u32 = 1000;

/// HTTP-backed [`ChainAdapter`] against a live Citrate JSON-RPC
/// endpoint.
///
/// Cheap to clone: the inner `reqwest::Client` + [`Wallet`] use
/// reference-counted handles internally.
#[derive(Clone)]
pub struct HttpChainAdapter {
    rpc_url: String,
    chain_id: u64,
    pool_contract: H160,
    wallet: Wallet,
    http: reqwest::Client,
    selectors: Selectors,
}

impl HttpChainAdapter {
    /// Construct an adapter pointed at `rpc_url`, signing writes to
    /// `pool_contract` with `wallet` on chain `chain_id`.
    pub fn new(
        rpc_url: String,
        chain_id: u64,
        pool_contract: H160,
        wallet: Wallet,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self {
            rpc_url,
            chain_id,
            pool_contract,
            wallet,
            http,
            selectors: Selectors::compute(),
        }
    }

    /// RM-B1 / WP-B2.4 (audit F-5): verify the RPC endpoint advertises
    /// the chain_id this adapter was configured with. Pre-fix, a
    /// pool-coordinator daemon could be launched against the wrong
    /// RPC (e.g., devnet vs testnet) and silently sign + submit
    /// transactions intended for the wrong chain — the network would
    /// reject them with cryptic errors but the daemon would keep
    /// trying.
    ///
    /// Issues `eth_chainId` once and compares the response to
    /// `self.chain_id`. On mismatch returns `CoordinatorError::Chain`
    /// with a descriptive message; the caller is expected to
    /// fail-fast (exit non-zero before entering the decision loop).
    pub async fn verify_rpc_chain_id(&self) -> Result<(), CoordinatorError> {
        let result = self.rpc("eth_chainId", json!([])).await?;
        let hex_str = result.as_str().ok_or_else(|| {
            CoordinatorError::Chain("eth_chainId result not a string".into())
        })?;
        let observed = parse_hex_u64(hex_str).map_err(|e| {
            CoordinatorError::Chain(format!("eth_chainId decode: {}", e))
        })?;
        if observed != self.chain_id {
            return Err(CoordinatorError::Chain(format!(
                "F-5: RPC chain_id mismatch — configured {} vs observed {} ({})",
                self.chain_id, observed, self.rpc_url
            )));
        }
        Ok(())
    }

    /// POST a JSON-RPC request and return the `result` field as a
    /// `serde_json::Value`. Errors returned by the node (non-null
    /// `error` field) map to `CoordinatorError::Chain`.
    async fn rpc(&self, method: &str, params: Value) -> Result<Value, CoordinatorError> {
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
            .map_err(|e| CoordinatorError::Chain(format!("{} transport: {}", method, e)))?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| CoordinatorError::Chain(format!("{} decode: {}", method, e)))?;
        if let Some(err) = value.get("error").filter(|v| !v.is_null()) {
            return Err(CoordinatorError::Chain(format!(
                "{} rpc error: {}",
                method, err
            )));
        }
        value
            .get("result")
            .cloned()
            .ok_or_else(|| CoordinatorError::Chain(format!("{} missing result", method)))
    }

    /// Issue `eth_call(to=pool_contract, data)` and return the
    /// decoded return bytes.
    async fn eth_call(&self, data: &[u8]) -> Result<Vec<u8>, CoordinatorError> {
        let to = format!("0x{}", hex::encode(self.pool_contract.as_bytes()));
        let data_hex = format!("0x{}", hex::encode(data));
        let params = json!([
            { "to": to, "data": data_hex },
            "latest",
        ]);
        let result = self.rpc("eth_call", params).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| CoordinatorError::Chain("eth_call result not a string".into()))?;
        hex::decode(hex_str.trim_start_matches("0x"))
            .map_err(|e| CoordinatorError::Chain(format!("eth_call bad hex: {}", e)))
    }

    /// Fetch the signer's nonce including pending-mempool txs.
    async fn fetch_nonce(&self) -> Result<u64, CoordinatorError> {
        let addr = format!("0x{}", hex::encode(self.wallet.address().as_bytes()));
        let result = self
            .rpc("eth_getTransactionCount", json!([addr, "pending"]))
            .await?;
        let hex_str = result.as_str().ok_or_else(|| {
            CoordinatorError::Chain("eth_getTransactionCount not a string".into())
        })?;
        parse_hex_u64(hex_str)
            .map_err(|e| CoordinatorError::Chain(format!("nonce decode: {}", e)))
    }

    /// Fetch the current gas price (legacy field — the Citrate node
    /// returns a reasonable base-fee estimate via `eth_gasPrice`).
    async fn fetch_gas_price(&self) -> Result<U256, CoordinatorError> {
        let result = self.rpc("eth_gasPrice", json!([])).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| CoordinatorError::Chain("eth_gasPrice not a string".into()))?;
        parse_hex_u256(hex_str)
            .map_err(|e| CoordinatorError::Chain(format!("gas price decode: {}", e)))
    }

    /// Build, sign, and submit an EIP-1559 write to
    /// `self.pool_contract` with the given calldata. Returns the
    /// transaction hash AND — if the tx confirmed within
    /// [`RECEIPT_WAIT_TIMEOUT`] — the mining block number.
    async fn send_write(
        &self,
        calldata: Vec<u8>,
    ) -> Result<(H256, Option<u64>), CoordinatorError> {
        let nonce = self.fetch_nonce().await?;
        let gas_price = self.fetch_gas_price().await?;
        let priority = U256::from(DEFAULT_PRIORITY_FEE_WEI);
        // Cap: EIP-1559 rule is max_fee ≥ max_priority_fee. We set
        // max_fee = 2 × base (gas_price) + priority as a generous
        // upper bound — the protocol refunds unused fees.
        let max_fee = gas_price.saturating_mul(U256::from(2u64)).saturating_add(priority);

        let tx = Eip1559Tx {
            chain_id: self.chain_id,
            nonce,
            max_priority_fee_per_gas: priority,
            max_fee_per_gas: max_fee,
            gas_limit: WRITE_GAS_LIMIT,
            to: self.pool_contract,
            value: U256::zero(),
            data: calldata,
        };
        let signed = self.wallet.sign_eip1559(&tx)?;
        let tx_hash = tx_hash_of_signed(&signed);
        let signed_hex = format!("0x{}", hex::encode(&signed));

        // Submit. We don't rely on the node echoing the hash —
        // the one we computed locally is authoritative.
        let _ = self
            .rpc("eth_sendRawTransaction", json!([signed_hex]))
            .await?;

        // Poll for receipt.
        let block_number = self.wait_for_receipt(tx_hash).await?;
        Ok((tx_hash, block_number))
    }

    /// Poll ComputePool for `ComputeRequested` events mined in
    /// `[from_block, to_block]`. Returns the decoded events in
    /// log-order (oldest first). Used by the daemon's event loop
    /// in main.rs.
    ///
    /// Event signature (indexed topics shown with ^):
    /// ```text
    /// ComputeRequested(
    ///   ^uint256 poolId,
    ///   ^uint256 jobId,
    ///   ^address requester,
    ///    uint256 payment
    /// )
    /// ```
    ///
    /// # Fields NOT decoded in this slice
    ///
    /// `prompt` and `max_tokens` require reading the job's stored
    /// `PoolJobSpec` bytes from chain and decoding the struct. That
    /// decode is non-trivial (mixed fixed + dynamic ABI) and lands
    /// in a follow-up when a live deployment surfaces a concrete
    /// test vector. For S0 the event carries empty prompt + the
    /// default max_tokens (1000) so handle_event's PoolInferRequest
    /// still composes; provider nodes must tolerate empty prompt
    /// during S0 bring-up.
    pub async fn poll_compute_requested(
        &self,
        from_block: u64,
        to_block: u64,
    ) -> Result<Vec<ComputeRequestedEvent>, CoordinatorError> {
        let event_sig = selector_full("ComputeRequested(uint256,uint256,address,uint256)");
        let event_sig_hex = format!("0x{}", hex::encode(event_sig));
        let address_hex = format!("0x{}", hex::encode(self.pool_contract.as_bytes()));
        let params = json!([{
            "fromBlock": format!("0x{:x}", from_block),
            "toBlock":   format!("0x{:x}", to_block),
            "address":   address_hex,
            "topics":    [event_sig_hex],
        }]);
        let result = self.rpc("eth_getLogs", params).await?;
        let arr = result
            .as_array()
            .ok_or_else(|| CoordinatorError::Chain("eth_getLogs not array".into()))?;

        let mut out = Vec::with_capacity(arr.len());
        for entry in arr {
            let topics = entry
                .get("topics")
                .and_then(|v| v.as_array())
                .ok_or_else(|| CoordinatorError::Chain("log missing topics".into()))?;
            if topics.len() < 4 {
                return Err(CoordinatorError::Chain(format!(
                    "log has {} topics, expected 4",
                    topics.len()
                )));
            }
            let pool_id = topic_as_u64(topics[1].as_str())?;
            let job_id = topic_as_u64(topics[2].as_str())?;
            let requester = topic_as_address(topics[3].as_str())?;

            // Non-indexed data: a single uint256 payment.
            let data_str = entry
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

            // Extract envelope fields for dedup across overlapping
            // polling windows. Missing fields are extremely unlikely
            // from a real node but we tolerate them defensively.
            let tx_hash = entry
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
            let log_index = entry
                .get("logIndex")
                .and_then(|v| v.as_str())
                .and_then(|s| parse_hex_u64(s).ok())
                .unwrap_or(0) as u32;
            let block_number = entry
                .get("blockNumber")
                .and_then(|v| v.as_str())
                .and_then(|s| parse_hex_u64(s).ok())
                .unwrap_or(0);

            // Fetch PoolJobSpec from chain to populate prompt +
            // max_tokens. One extra eth_call per event; the happy
            // path is under tens of events per poll window, so this
            // is cheap. Failures fall back to empty prompt + default
            // max_tokens — a log, not a fatal error.
            let (prompt, max_tokens) = match self.fetch_job_spec(job_id).await {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(
                        job_id = job_id,
                        error = %e,
                        "fetch_job_spec failed; dispatching with defaults"
                    );
                    (String::new(), DEFAULT_MAX_TOKENS)
                }
            };

            out.push(ComputeRequestedEvent {
                pool_id,
                job_id,
                requester,
                payment_grains: payment,
                prompt,
                max_tokens,
                tx_hash,
                log_index,
                block_number,
            });
        }
        Ok(out)
    }

    /// Fetch a job's `PoolJobSpec` from chain storage and return the
    /// fields `handle_event` needs for the HTTPS dispatch body:
    /// `(prompt, max_tokens)`.
    ///
    /// Data source (Rule 11):
    ///   `ComputePool.getJob(uint256)` → PoolJob tuple; the 3rd
    ///   field is `bytes jobSpec`. That bytes payload is itself an
    ///   `abi.encode(PoolJobSpec)` — we parse both levels inline.
    ///
    /// Returns defaults (empty prompt, DEFAULT_MAX_TOKENS) if the
    /// job has no jobSpec bytes stored (legacy callers using the
    /// opaque `requestPoolCompute(bytes)` path may not populate it).
    pub async fn fetch_job_spec(
        &self,
        job_id: u64,
    ) -> Result<(String, u32), CoordinatorError> {
        // Step 1: getJob(jobId) → PoolJob memory.
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.get_job);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        let ret = self.eth_call(&data).await?;

        // Step 2: PoolJob has 8 fields — head is 8 × 32 = 256 bytes.
        // Layout:
        //   [  0..  32] poolId (uint256)
        //   [ 32..  64] requester (address, padded)
        //   [ 64..  96] offset to jobSpec bytes (dynamic)
        //   [ 96.. 128] payment (uint256)
        //   [128.. 160] status (uint8, padded)
        //   [160.. 192] createdAt (uint256)
        //   [192.. 224] dispatchBlock (uint256)
        //   [224.. 256] dispatchedBy (address, padded)
        if ret.len() < 256 {
            return Err(CoordinatorError::Chain(format!(
                "PoolJob return too short: {} bytes",
                ret.len()
            )));
        }
        let jobspec_bytes = decode_dynamic_bytes_at_offset(&ret, 64)?;

        // Legacy jobs may omit PoolJobSpec — return defaults rather
        // than refuse to dispatch.
        if jobspec_bytes.is_empty() {
            return Ok((String::new(), DEFAULT_MAX_TOKENS));
        }

        // Step 3: PoolJobSpec has 7 fields — head is 7 × 32 = 224 bytes.
        // Layout:
        //   [  0..  32] version (uint8, padded)
        //   [ 32..  64] mode (uint8, padded)
        //   [ 64..  96] modelHash (bytes32)
        //   [ 96.. 128] offset to inputData bytes (dynamic)
        //   [128.. 160] maxTokens (uint32, padded)
        //   [160.. 192] verificationTier (uint8, padded)
        //   [192.. 224] batchSize (uint32, padded)
        if jobspec_bytes.len() < 224 {
            return Err(CoordinatorError::Chain(format!(
                "PoolJobSpec too short: {} bytes",
                jobspec_bytes.len()
            )));
        }
        let max_tokens_u256 = U256::from_big_endian(&jobspec_bytes[128..160]);
        let max_tokens = if max_tokens_u256 > U256::from(u32::MAX) {
            u32::MAX
        } else {
            let v = max_tokens_u256.as_u32();
            if v == 0 { DEFAULT_MAX_TOKENS } else { v }
        };
        let input_bytes = decode_dynamic_bytes_at_offset(&jobspec_bytes, 96)?;

        // Convert to UTF-8 with lossy replacement — chains may store
        // binary content (e.g. an IPFS CID encoded as bytes); lossy
        // keeps handle_event from failing on otherwise-valid jobs.
        let prompt = String::from_utf8_lossy(&input_bytes).to_string();
        Ok((prompt, max_tokens))
    }

    /// Fetch the latest block number from the node (for the event
    /// loop's `to_block` bound on each poll).
    pub async fn latest_block(&self) -> Result<u64, CoordinatorError> {
        let result = self.rpc("eth_blockNumber", json!([])).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| CoordinatorError::Chain("eth_blockNumber not a string".into()))?;
        parse_hex_u64(hex_str)
            .map_err(|e| CoordinatorError::Chain(format!("blockNumber decode: {}", e)))
    }

    /// Poll `eth_getTransactionReceipt` with exponential backoff
    /// until we see a non-null receipt, then extract its block
    /// number. Returns `Ok(None)` on timeout (the tx MAY still mine
    /// later; the caller treats this as a best-effort submit).
    async fn wait_for_receipt(&self, tx_hash: H256) -> Result<Option<u64>, CoordinatorError> {
        let hash_hex = format!("0x{}", hex::encode(tx_hash.as_bytes()));
        let deadline = tokio::time::Instant::now() + RECEIPT_WAIT_TIMEOUT;
        let mut delay = RECEIPT_POLL_INITIAL;

        while tokio::time::Instant::now() < deadline {
            let result = self
                .rpc("eth_getTransactionReceipt", json!([hash_hex]))
                .await?;
            if !result.is_null() {
                // Extract status + blockNumber. The node returns
                // status=0x1 on success, 0x0 on revert. We treat a
                // reverted tx as a chain error — the caller logs and
                // the on-chain state hasn't changed.
                let status = result
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0x1");
                if parse_hex_u64(status).unwrap_or(1) == 0 {
                    return Err(CoordinatorError::Chain(format!(
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
        // Timed out waiting. The tx may yet mine; surface None so
        // the caller can still log the tx hash for follow-up.
        tracing::warn!(tx_hash = ?tx_hash, "receipt not found within timeout");
        Ok(None)
    }
}

#[async_trait]
impl ChainAdapter for HttpChainAdapter {
    fn self_address(&self) -> H160 {
        self.wallet.address()
    }

    async fn coordinator_for(
        &self,
        pool_id: u64,
        epoch: u64,
    ) -> Result<H160, CoordinatorError> {
        // coordinatorFor(uint256 poolId, uint256 epoch) → (address)
        let mut data = Vec::with_capacity(4 + 64);
        data.extend_from_slice(&self.selectors.coordinator_for);
        data.extend_from_slice(&u256_word(U256::from(pool_id)));
        data.extend_from_slice(&u256_word(U256::from(epoch)));
        let ret = self.eth_call(&data).await?;
        if ret.len() < 32 {
            return Err(CoordinatorError::Chain(format!(
                "coordinatorFor return too short: {} bytes",
                ret.len()
            )));
        }
        // Address is last 20 of the 32-byte word.
        Ok(H160::from_slice(&ret[12..32]))
    }

    async fn pool_members(
        &self,
        pool_id: u64,
    ) -> Result<Vec<PoolMemberInfo>, CoordinatorError> {
        // Step 1: getPoolMembers(uint256) → address[]
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.get_pool_members);
        data.extend_from_slice(&u256_word(U256::from(pool_id)));
        let ret = self.eth_call(&data).await?;
        let addrs = decode_address_array(&ret)?;

        // Step 2: getMember(poolId, address) per address. Sequential
        // to keep the implementation simple; a pool with hundreds of
        // members could parallelise via `join_all` — left as a future
        // perf pass.
        let mut out = Vec::with_capacity(addrs.len());
        for addr in addrs {
            let mut call = Vec::with_capacity(4 + 64);
            call.extend_from_slice(&self.selectors.get_member);
            call.extend_from_slice(&u256_word(U256::from(pool_id)));
            call.extend_from_slice(&pad_address_left(addr));
            let ret = self.eth_call(&call).await?;
            let (gpu_count, active) = decode_member_struct(&ret)?;
            out.push(PoolMemberInfo {
                address: addr,
                gpu_count,
                active,
            });
        }
        Ok(out)
    }

    async fn record_dispatch(
        &self,
        job_id: u64,
        _member: H160,
    ) -> Result<RecordDispatchOutcome, CoordinatorError> {
        // recordDispatch(uint256 jobId). `member` isn't a contract
        // argument — it's carried via the off-chain dispatch path
        // and recorded in logs by the daemon caller.
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.record_dispatch);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        let (tx_hash, block_number) = self.send_write(data).await?;
        Ok(RecordDispatchOutcome::Confirmed {
            tx_hash,
            // Unknown block number on receipt timeout is surfaced as
            // block 0 — the caller's downstream log uses tx_hash as
            // the primary identifier anyway.
            block_number: block_number.unwrap_or(0),
        })
    }

    async fn complete_job(&self, job_id: u64) -> Result<H256, CoordinatorError> {
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.complete_job);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        let (tx_hash, _) = self.send_write(data).await?;
        Ok(tx_hash)
    }

    async fn fail_job(&self, job_id: u64) -> Result<H256, CoordinatorError> {
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.fail_job);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        let (tx_hash, _) = self.send_write(data).await?;
        Ok(tx_hash)
    }
}

// ── ABI helpers (private) ────────────────────────────────────────

fn selector(sig: &str) -> [u8; 4] {
    let mut h = Keccak256::new();
    h.update(sig.as_bytes());
    let out = h.finalize();
    [out[0], out[1], out[2], out[3]]
}

/// Full 32-byte keccak — used for event topic hashes, where the
/// whole digest (not just 4-byte selector) is the indexed topic.
fn selector_full(sig: &str) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(sig.as_bytes());
    let out = h.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    buf
}

fn topic_as_u64(topic: Option<&str>) -> Result<u64, CoordinatorError> {
    let s = topic.ok_or_else(|| CoordinatorError::Chain("topic missing".into()))?;
    let stripped = s.trim_start_matches("0x");
    if stripped.len() != 64 {
        return Err(CoordinatorError::Chain(format!(
            "topic has {} hex chars, expected 64",
            stripped.len()
        )));
    }
    let bytes = hex::decode(stripped)
        .map_err(|e| CoordinatorError::Chain(format!("topic hex: {}", e)))?;
    Ok(U256::from_big_endian(&bytes).as_u64())
}

fn topic_as_address(topic: Option<&str>) -> Result<H160, CoordinatorError> {
    let s = topic.ok_or_else(|| CoordinatorError::Chain("topic missing".into()))?;
    let stripped = s.trim_start_matches("0x");
    if stripped.len() != 64 {
        return Err(CoordinatorError::Chain(format!(
            "address topic has {} hex chars, expected 64",
            stripped.len()
        )));
    }
    let bytes = hex::decode(stripped)
        .map_err(|e| CoordinatorError::Chain(format!("address topic hex: {}", e)))?;
    Ok(H160::from_slice(&bytes[12..32]))
}

fn u256_word(v: U256) -> [u8; 32] {
    let mut buf = [0u8; 32];
    v.to_big_endian(&mut buf);
    buf
}

fn pad_address_left(addr: H160) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[12..32].copy_from_slice(addr.as_bytes());
    buf
}

/// Decode `address[]` ABI return: 32-byte offset (= 0x20) + 32-byte
/// length + N × 32-byte left-padded addresses.
fn decode_address_array(bytes: &[u8]) -> Result<Vec<H160>, CoordinatorError> {
    if bytes.len() < 64 {
        return Err(CoordinatorError::Chain(format!(
            "address[] return too short: {} bytes",
            bytes.len()
        )));
    }
    let len = U256::from_big_endian(&bytes[32..64]).as_usize();
    let expected = 64 + len * 32;
    if bytes.len() < expected {
        return Err(CoordinatorError::Chain(format!(
            "address[] truncated: len={} need {} bytes have {}",
            len,
            expected,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let start = 64 + i * 32;
        let word = &bytes[start..start + 32];
        out.push(H160::from_slice(&word[12..32]));
    }
    Ok(out)
}

/// Decode the `PoolMember` tuple returned by `getMember(poolId, addr)`:
///
///   (uint256 gpuCount, uint256 stake, uint256 activeJobs,
///    uint256 joinedAt, bool active)
///
/// All fields are 32 bytes packed, total 160 bytes. We only return
/// `gpuCount` (as u32) and `active` — the caller discards the rest.
/// `gpuCount > u32::MAX` is clamped to `u32::MAX`; a provider with
/// more than 4B GPUs is not physically realistic.
fn decode_member_struct(bytes: &[u8]) -> Result<(u32, bool), CoordinatorError> {
    if bytes.len() < 160 {
        return Err(CoordinatorError::Chain(format!(
            "PoolMember return too short: {} bytes",
            bytes.len()
        )));
    }
    let gpu_count_u256 = U256::from_big_endian(&bytes[0..32]);
    let gpu_count = if gpu_count_u256 > U256::from(u32::MAX) {
        u32::MAX
    } else {
        gpu_count_u256.as_u32()
    };
    // bool sits in the last byte of its 32-byte word.
    let active = bytes[159] != 0;
    Ok((gpu_count, active))
}

/// Decode an ABI-encoded `bytes` field at the given offset-word
/// position. `buf` is the full return payload; `offset_word_pos` is
/// the byte position of the 32-byte word containing the offset.
/// Used for nested dynamic `bytes` fields inside struct-returning
/// getters.
fn decode_dynamic_bytes_at_offset(
    buf: &[u8],
    offset_word_pos: usize,
) -> Result<Vec<u8>, CoordinatorError> {
    if buf.len() < offset_word_pos + 32 {
        return Err(CoordinatorError::Chain(format!(
            "offset word at {} out of range (buf={})",
            offset_word_pos,
            buf.len()
        )));
    }
    let offset =
        U256::from_big_endian(&buf[offset_word_pos..offset_word_pos + 32]).as_usize();
    if buf.len() < offset + 32 {
        return Err(CoordinatorError::Chain(format!(
            "dynamic bytes length word at {} out of range (buf={})",
            offset,
            buf.len()
        )));
    }
    let len = U256::from_big_endian(&buf[offset..offset + 32]).as_usize();
    let data_start = offset + 32;
    if buf.len() < data_start + len {
        return Err(CoordinatorError::Chain(format!(
            "dynamic bytes data truncated: len={} buf={}",
            len,
            buf.len()
        )));
    }
    Ok(buf[data_start..data_start + len].to_vec())
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

    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::routing::post;
    use axum::Json;
    use serde_json::Value;
    use tokio::net::TcpListener;

    /// Well-known key; derives 0x7e5f4552091a69125d5dfcb7b8c2659029395bdf.
    const TEST_HEX: &str =
        "0000000000000000000000000000000000000000000000000000000000000001";

    /// Records every JSON-RPC request the stub server saw so tests can
    /// assert on method + params.
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

        fn last_params(&self, method: &str) -> Option<Value> {
            self.calls
                .lock()
                .expect("rpc log poisoned")
                .iter()
                .rev()
                .find(|(m, _)| m == method)
                .map(|(_, p)| p.clone())
        }
    }

    /// Stub server state: per-method canned response map. Each call
    /// pops the next response off the VecDeque so tests can script a
    /// sequence (e.g. "receipt null, then non-null").
    #[derive(Clone)]
    struct StubState {
        log: RpcLog,
        responses: Arc<Mutex<std::collections::HashMap<String, std::collections::VecDeque<Value>>>>,
    }

    impl StubState {
        fn new() -> Self {
            Self {
                log: RpcLog::default(),
                responses: Arc::new(Mutex::new(std::collections::HashMap::new())),
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

    fn pool_contract() -> H160 {
        H160::repeat_byte(0xCC)
    }

    fn make_adapter(rpc_url: String) -> HttpChainAdapter {
        let wallet = Wallet::from_hex(TEST_HEX).expect("wallet");
        HttpChainAdapter::new(rpc_url, 40204, pool_contract(), wallet)
    }

    /// Build an `address[]` ABI-encoded payload for a single address.
    fn encode_address_array_one(addr: H160) -> String {
        let mut buf = vec![0u8; 128];
        // offset = 0x20
        buf[31] = 0x20;
        // length = 1
        buf[63] = 1;
        // address in last 20 bytes of the word at [64..96]
        buf[64 + 12..64 + 32].copy_from_slice(addr.as_bytes());
        format!("0x{}", hex::encode(buf))
    }

    /// Build a `PoolMember` tuple with the given gpuCount and active.
    fn encode_member_tuple(gpu_count: u32, active: bool) -> String {
        let mut buf = vec![0u8; 160];
        U256::from(gpu_count).to_big_endian(&mut buf[0..32]);
        // stake, activeJobs, joinedAt left zero
        buf[159] = if active { 1 } else { 0 };
        format!("0x{}", hex::encode(buf))
    }

    /// Left-pad an address into a 32-byte hex word (the shape
    /// `coordinatorFor` returns).
    fn encode_address_word(addr: H160) -> String {
        let mut buf = vec![0u8; 32];
        buf[12..32].copy_from_slice(addr.as_bytes());
        format!("0x{}", hex::encode(buf))
    }

    // ── Tests ────────────────────────────────────────────────────

    #[test]
    fn selectors_are_distinct_and_four_bytes() {
        let s = Selectors::compute();
        let all = [
            s.coordinator_for,
            s.get_pool_members,
            s.get_member,
            s.record_dispatch,
            s.complete_job,
            s.fail_job,
        ];
        // All 4 bytes (trivially true by type).
        assert_eq!(all[0].len(), 4);
        // Pairwise distinct — a copy-paste bug in the signatures
        // would produce collisions here.
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "selector collision at {} vs {}", i, j);
            }
        }
    }

    #[test]
    fn parse_hex_u64_handles_prefixes_and_empty() {
        assert_eq!(parse_hex_u64("0x0").expect("parse"), 0);
        assert_eq!(parse_hex_u64("0x").expect("parse"), 0);
        assert_eq!(parse_hex_u64("0x10").expect("parse"), 16);
        assert_eq!(parse_hex_u64("1f").expect("parse"), 31);
        assert!(parse_hex_u64("0xzz").is_err());
    }

    #[test]
    fn decode_member_struct_extracts_gpu_count_and_active() {
        // gpuCount=3 active=true
        let mut buf = vec![0u8; 160];
        U256::from(3u64).to_big_endian(&mut buf[0..32]);
        buf[159] = 1;
        let (gpu, active) = decode_member_struct(&buf).expect("decode");
        assert_eq!(gpu, 3);
        assert!(active);

        // gpuCount=0 active=false
        let buf2 = vec![0u8; 160];
        let (gpu2, active2) = decode_member_struct(&buf2).expect("decode");
        assert_eq!(gpu2, 0);
        assert!(!active2);
    }

    #[test]
    fn decode_member_struct_rejects_short() {
        let short = vec![0u8; 159];
        assert!(matches!(
            decode_member_struct(&short).unwrap_err(),
            CoordinatorError::Chain(_)
        ));
    }

    #[tokio::test]
    async fn coordinator_for_decodes_address() {
        let state = StubState::new();
        let elected = H160::repeat_byte(0xAB);
        state.queue("eth_call", json!(encode_address_word(elected)));
        let addr = spawn_stub_rpc(state.clone()).await;

        let adapter = make_adapter(format!("http://{}", addr));
        let got = adapter.coordinator_for(1, 0).await.expect("call");

        assert_eq!(got, elected);
        // Verify it actually went through eth_call with our calldata
        // starting with the coordinatorFor selector.
        let params = state.log.last_params("eth_call").expect("saw call");
        let data_hex = params[0]["data"].as_str().expect("data present");
        let data = hex::decode(data_hex.trim_start_matches("0x")).expect("decode");
        let sel = Selectors::compute().coordinator_for;
        assert_eq!(&data[0..4], &sel);
        // pool_id=1 in the first uint256
        let pool_word = U256::from_big_endian(&data[4..36]);
        assert_eq!(pool_word, U256::from(1u64));
        // epoch=0 in the second uint256
        let epoch_word = U256::from_big_endian(&data[36..68]);
        assert_eq!(epoch_word, U256::zero());
    }

    #[tokio::test]
    async fn coordinator_for_rejects_short_return() {
        let state = StubState::new();
        state.queue("eth_call", json!("0xdeadbeef"));
        let addr = spawn_stub_rpc(state).await;

        let adapter = make_adapter(format!("http://{}", addr));
        let err = adapter.coordinator_for(1, 0).await.expect_err("should fail");
        assert!(matches!(err, CoordinatorError::Chain(_)));
    }

    #[tokio::test]
    async fn pool_members_decodes_single_member() {
        let state = StubState::new();
        let member = H160::repeat_byte(0x11);
        // First eth_call is getPoolMembers → address[] with one entry.
        state.queue("eth_call", json!(encode_address_array_one(member)));
        // Second eth_call is getMember → tuple with gpuCount=4 active=true.
        state.queue("eth_call", json!(encode_member_tuple(4, true)));
        let addr = spawn_stub_rpc(state).await;

        let adapter = make_adapter(format!("http://{}", addr));
        let members = adapter.pool_members(7).await.expect("call");

        assert_eq!(members.len(), 1);
        assert_eq!(members[0].address, member);
        assert_eq!(members[0].gpu_count, 4);
        assert!(members[0].active);
    }

    #[tokio::test]
    async fn record_dispatch_signs_and_submits_eip1559() {
        let state = StubState::new();
        // Daemon fetches nonce, gas price, submits raw tx, polls receipt.
        state.queue("eth_getTransactionCount", json!("0x5"));
        state.queue("eth_gasPrice", json!("0x3b9aca00")); // 1 gwei
        state.queue("eth_sendRawTransaction", json!("0xdeadbeef"));
        // Receipt comes back on the first poll.
        state.queue(
            "eth_getTransactionReceipt",
            json!({
                "status": "0x1",
                "blockNumber": "0x2a",
            }),
        );
        let addr = spawn_stub_rpc(state.clone()).await;

        let adapter = make_adapter(format!("http://{}", addr));
        let outcome = adapter
            .record_dispatch(42, H160::repeat_byte(0x11))
            .await
            .expect("call");

        let RecordDispatchOutcome::Confirmed {
            tx_hash: _,
            block_number,
        } = outcome;
        assert_eq!(block_number, 0x2a);

        // Check the sendRawTransaction payload was a 0x02-prefixed
        // EIP-1559 envelope.
        let params = state
            .log
            .last_params("eth_sendRawTransaction")
            .expect("saw send");
        let raw_hex = params[0]
            .as_str()
            .expect("raw tx is string")
            .to_string();
        let raw = hex::decode(raw_hex.trim_start_matches("0x")).expect("decode raw");
        assert_eq!(
            raw[0], 0x02,
            "EIP-1559 type prefix missing from submitted tx"
        );

        // Verify all the expected methods were called in order.
        let methods = state.log.methods();
        assert_eq!(methods[0], "eth_getTransactionCount");
        assert_eq!(methods[1], "eth_gasPrice");
        assert_eq!(methods[2], "eth_sendRawTransaction");
        assert_eq!(methods[3], "eth_getTransactionReceipt");
    }

    #[tokio::test]
    async fn complete_job_polls_until_receipt_arrives() {
        let state = StubState::new();
        state.queue("eth_getTransactionCount", json!("0x1"));
        state.queue("eth_gasPrice", json!("0x3b9aca00"));
        state.queue("eth_sendRawTransaction", json!("0x00"));
        // First receipt poll returns null, second returns success.
        state.queue("eth_getTransactionReceipt", Value::Null);
        state.queue(
            "eth_getTransactionReceipt",
            json!({
                "status": "0x1",
                "blockNumber": "0x7",
            }),
        );
        let addr = spawn_stub_rpc(state.clone()).await;

        let adapter = make_adapter(format!("http://{}", addr));
        let tx_hash = adapter.complete_job(99).await.expect("call");

        // tx_hash should be non-zero (derived from signed RLP).
        assert_ne!(tx_hash, H256::zero());

        // We should have two receipt polls.
        let receipt_polls = state
            .log
            .methods()
            .iter()
            .filter(|m| *m == "eth_getTransactionReceipt")
            .count();
        assert_eq!(receipt_polls, 2);
    }

    #[tokio::test]
    async fn fail_job_surfaces_reverted_receipt() {
        let state = StubState::new();
        state.queue("eth_getTransactionCount", json!("0x2"));
        state.queue("eth_gasPrice", json!("0x3b9aca00"));
        state.queue("eth_sendRawTransaction", json!("0x00"));
        state.queue(
            "eth_getTransactionReceipt",
            json!({
                "status": "0x0",
                "blockNumber": "0x9",
            }),
        );
        let addr = spawn_stub_rpc(state).await;

        let adapter = make_adapter(format!("http://{}", addr));
        let err = adapter.fail_job(13).await.expect_err("should error");
        assert!(matches!(err, CoordinatorError::Chain(_)));
    }

    #[tokio::test]
    async fn rpc_errors_propagate_as_chain_error() {
        // Stub returns a JSON-RPC error body.
        let state = StubState::new();
        let addr = {
            let state = state.clone();
            let app = axum::Router::new()
                .route(
                    "/",
                    post(
                        |State(_): State<StubState>, Json(body): Json<Value>| async move {
                            let id = body.get("id").cloned().unwrap_or(json!(1));
                            Json(json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": { "code": -32000, "message": "stub failure" },
                            }))
                        },
                    ),
                )
                .with_state(state);
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("local_addr");
            tokio::spawn(async move {
                axum::serve(listener, app).await.expect("serve");
            });
            addr
        };

        let adapter = make_adapter(format!("http://{}", addr));
        let err = adapter.coordinator_for(1, 0).await.expect_err("fail");
        assert!(matches!(err, CoordinatorError::Chain(_)));
    }

    #[test]
    fn self_address_returns_wallet_address() {
        let wallet = Wallet::from_hex(TEST_HEX).expect("wallet");
        let expected = wallet.address();
        let adapter = HttpChainAdapter::new(
            "http://unused".into(),
            40204,
            pool_contract(),
            wallet,
        );
        assert_eq!(adapter.self_address(), expected);
    }

    // ── Event polling (poll_compute_requested) ──────────────────

    fn pad_u64_topic(value: u64) -> String {
        let mut buf = [0u8; 32];
        U256::from(value).to_big_endian(&mut buf);
        format!("0x{}", hex::encode(buf))
    }

    fn pad_address_topic(addr: H160) -> String {
        let mut buf = [0u8; 32];
        buf[12..32].copy_from_slice(addr.as_bytes());
        format!("0x{}", hex::encode(buf))
    }

    fn compute_requested_sig_hex() -> String {
        let sig =
            selector_full("ComputeRequested(uint256,uint256,address,uint256)");
        format!("0x{}", hex::encode(sig))
    }

    #[tokio::test]
    async fn poll_compute_requested_decodes_one_log() {
        let state = StubState::new();
        let requester = H160::repeat_byte(0xAB);
        let payment = 123_000_000_000_000_000u64; // 0.123 ether
        let tx_hash = H256::repeat_byte(0xEE);

        // Compose a ComputeRequested log: topics[0]=sig, topics[1..4]=
        // indexed fields, data=payment.
        let mut data_buf = [0u8; 32];
        U256::from(payment).to_big_endian(&mut data_buf);
        let log = json!({
            "address": format!("0x{}", hex::encode(pool_contract().as_bytes())),
            "topics": [
                compute_requested_sig_hex(),
                pad_u64_topic(7),    // poolId
                pad_u64_topic(42),   // jobId
                pad_address_topic(requester),
            ],
            "data": format!("0x{}", hex::encode(data_buf)),
            "blockNumber": "0x10",
            "transactionHash": format!("0x{}", hex::encode(tx_hash.as_bytes())),
            "logIndex": "0x3",
        });
        state.queue("eth_getLogs", json!([log]));

        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let events = adapter
            .poll_compute_requested(0, 100)
            .await
            .expect("poll events");

        assert_eq!(events.len(), 1);
        let ev = &events[0];
        assert_eq!(ev.pool_id, 7);
        assert_eq!(ev.job_id, 42);
        assert_eq!(ev.requester, requester);
        assert_eq!(ev.payment_grains, U256::from(payment));
        // fetch_job_spec falls back (no queued response) → defaults.
        assert_eq!(ev.prompt, "");
        assert_eq!(ev.max_tokens, DEFAULT_MAX_TOKENS);
        // Envelope fields populated for reorg-dedup.
        assert_eq!(ev.tx_hash, tx_hash);
        assert_eq!(ev.log_index, 3);
        assert_eq!(ev.block_number, 0x10);
    }

    #[tokio::test]
    async fn poll_compute_requested_rejects_short_topics() {
        let state = StubState::new();
        // Only 2 topics instead of 4 — must reject.
        let log = json!({
            "address": format!("0x{}", hex::encode(pool_contract().as_bytes())),
            "topics": [
                compute_requested_sig_hex(),
                pad_u64_topic(1),
            ],
            "data": "0x",
            "blockNumber": "0x1",
        });
        state.queue("eth_getLogs", json!([log]));

        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let err = adapter
            .poll_compute_requested(0, 10)
            .await
            .expect_err("must reject short topics");
        assert!(matches!(err, CoordinatorError::Chain(_)));
    }

    // ── fetch_job_spec (PoolJobSpec decoding) ───────────────────

    /// Build an ABI-encoded PoolJobSpec bytes payload given
    /// (version, mode, modelHash, inputData, maxTokens,
    /// verificationTier, batchSize).
    fn encode_poolJobSpec_bytes(
        version: u8,
        mode: u8,
        model_hash: [u8; 32],
        input_data: &[u8],
        max_tokens: u32,
        verification_tier: u8,
        batch_size: u32,
    ) -> Vec<u8> {
        // Head: 7 × 32 bytes = 224.
        let input_offset_word = 224u64; // head_size; dynamic tail starts here
        let mut out = Vec::new();
        // word 0: version
        let mut w = [0u8; 32];
        w[31] = version;
        out.extend_from_slice(&w);
        // word 1: mode
        let mut w = [0u8; 32];
        w[31] = mode;
        out.extend_from_slice(&w);
        // word 2: modelHash (already 32 bytes)
        out.extend_from_slice(&model_hash);
        // word 3: offset to inputData
        let mut w = [0u8; 32];
        U256::from(input_offset_word).to_big_endian(&mut w);
        out.extend_from_slice(&w);
        // word 4: maxTokens
        let mut w = [0u8; 32];
        U256::from(max_tokens).to_big_endian(&mut w);
        out.extend_from_slice(&w);
        // word 5: verificationTier
        let mut w = [0u8; 32];
        w[31] = verification_tier;
        out.extend_from_slice(&w);
        // word 6: batchSize
        let mut w = [0u8; 32];
        U256::from(batch_size).to_big_endian(&mut w);
        out.extend_from_slice(&w);
        // Tail: length-prefix + data + zero-pad to 32
        let mut len_word = [0u8; 32];
        U256::from(input_data.len() as u64).to_big_endian(&mut len_word);
        out.extend_from_slice(&len_word);
        out.extend_from_slice(input_data);
        let pad = (32 - (input_data.len() % 32)) % 32;
        out.extend_from_slice(&vec![0u8; pad]);
        out
    }

    /// Build an ABI-encoded PoolJob return given the fields we
    /// care about (jobspec_bytes payload; other fields are zeros).
    fn encode_poolJob_return(jobspec_bytes: &[u8]) -> String {
        // Head: 8 × 32 = 256 bytes. jobSpec offset points PAST the
        // head.
        let jobspec_offset_word = 256u64;
        let mut out = Vec::new();
        // words 0, 1: poolId, requester — zeros
        out.extend_from_slice(&[0u8; 32]);
        out.extend_from_slice(&[0u8; 32]);
        // word 2: jobSpec offset
        let mut w = [0u8; 32];
        U256::from(jobspec_offset_word).to_big_endian(&mut w);
        out.extend_from_slice(&w);
        // words 3..=7: payment, status, createdAt, dispatchBlock, dispatchedBy — zeros
        out.extend_from_slice(&[0u8; 32 * 5]);
        // Tail: length prefix + data + pad
        let mut len_word = [0u8; 32];
        U256::from(jobspec_bytes.len() as u64).to_big_endian(&mut len_word);
        out.extend_from_slice(&len_word);
        out.extend_from_slice(jobspec_bytes);
        let pad = (32 - (jobspec_bytes.len() % 32)) % 32;
        out.extend_from_slice(&vec![0u8; pad]);
        format!("0x{}", hex::encode(out))
    }

    #[tokio::test]
    async fn fetch_job_spec_decodes_prompt_and_max_tokens() {
        let state = StubState::new();
        let prompt_bytes = b"Explain pipeline parallelism.";
        let inner = encode_poolJobSpec_bytes(
            1,                      // version
            0,                      // mode = InferencePool
            [0xAB; 32],             // modelHash
            prompt_bytes,           // inputData
            256,                    // maxTokens
            0,                      // verificationTier
            1,                      // batchSize
        );
        state.queue("eth_call", json!(encode_poolJob_return(&inner)));

        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let (prompt, max_tokens) =
            adapter.fetch_job_spec(42).await.expect("fetch");
        assert_eq!(prompt, "Explain pipeline parallelism.");
        assert_eq!(max_tokens, 256);
    }

    #[tokio::test]
    async fn fetch_job_spec_returns_defaults_on_empty_jobspec() {
        let state = StubState::new();
        // PoolJob with zero-length jobSpec bytes — defaults expected.
        state.queue("eth_call", json!(encode_poolJob_return(&[])));

        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let (prompt, max_tokens) =
            adapter.fetch_job_spec(1).await.expect("fetch");
        assert_eq!(prompt, "");
        assert_eq!(max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[tokio::test]
    async fn fetch_job_spec_rejects_short_return() {
        let state = StubState::new();
        // Return is too short to hold even the 8-word head.
        state.queue("eth_call", json!("0x1234"));
        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let err = adapter.fetch_job_spec(1).await.expect_err("reject");
        assert!(matches!(err, CoordinatorError::Chain(_)));
    }

    #[tokio::test]
    async fn latest_block_returns_u64() {
        let state = StubState::new();
        state.queue("eth_blockNumber", json!("0x2a")); // 42
        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let latest = adapter.latest_block().await.expect("latest");
        assert_eq!(latest, 42);
    }

    // ────────────────────────────────────────────────────────────
    // F-5 (RM-B1 / WP-B2.4): verify_rpc_chain_id regression tests.
    // ────────────────────────────────────────────────────────────

    /// F-5.1: matching chain_id passes verification.
    #[tokio::test]
    async fn f5_verify_chain_id_accepts_match() {
        let state = StubState::new();
        // 40204 == 0x9D0C
        state.queue("eth_chainId", json!("0x9D0C"));
        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        adapter
            .verify_rpc_chain_id()
            .await
            .expect("F-5: matching chain_id must verify");
    }

    /// F-5.2: mismatching chain_id is rejected with a descriptive
    /// error. Pre-fix the daemon would silently start signing
    /// against the wrong chain.
    #[tokio::test]
    async fn f5_verify_chain_id_rejects_mismatch() {
        let state = StubState::new();
        // Adapter is configured for 40204 (0x9D0C); RPC reports 1
        // (0x1, mainnet).
        state.queue("eth_chainId", json!("0x1"));
        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        let err = adapter
            .verify_rpc_chain_id()
            .await
            .expect_err("F-5: mismatched chain_id must reject");
        let msg = format!("{}", err);
        assert!(
            msg.contains("F-5") && msg.contains("40204") && msg.contains("1"),
            "F-5: error must name configured + observed chain_ids; got {}",
            msg
        );
    }

    /// F-5.3: an RPC that returns garbage for `eth_chainId` (string
    /// that doesn't decode as hex) is rejected.
    #[tokio::test]
    async fn f5_verify_chain_id_rejects_malformed_response() {
        let state = StubState::new();
        state.queue("eth_chainId", json!("not-a-hex-string"));
        let addr = spawn_stub_rpc(state).await;
        let adapter = make_adapter(format!("http://{}", addr));
        adapter
            .verify_rpc_chain_id()
            .await
            .expect_err("F-5: malformed chain_id response must reject");
    }
}
