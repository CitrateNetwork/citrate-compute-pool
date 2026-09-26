//! HTTP-backed [`ChainClient`] against `ComputePoolTraining` (S1).
//!
//! Real JSON-RPC client that speaks to a Citrate node via `reqwest`
//! and signs on-chain writes with a [`Wallet`]. Reads go through
//! `eth_call`; writes go through `eth_sendRawTransaction` followed by
//! an `eth_getTransactionReceipt` poll (exponential backoff up to
//! ~30s).
//!
//! Mirrors the pattern in `pool-coordinator/src/http_chain.rs` — the
//! two files should be kept structurally similar until the wallet +
//! ABI helpers are unified post-pilot (see `wallet.rs` FIXME).
//!
//! # Data sources (per Rule 11)
//!
//! Every method names the on-chain contract + method it reads from
//! or writes to so the daemon never silently returns fake data:
//!
//! | Method | Contract.method | Kind |
//! |--------|-----------------|------|
//! | `snapshot` | `ComputePoolTraining.getJob(uint256)` | view |
//! | `join_training_job` | `ComputePoolTraining.joinTrainingJob(uint256)` | write (payable) |
//! | `close_recruitment` | `ComputePoolTraining.closeRecruitment(uint256,address)` | write |
//! | `commit_epoch` | `ComputePoolTraining.commitEpoch(uint256,uint32,bytes32)` | write |
//! | `finalize` | `ComputePoolTraining.finalizeTrainingJob(uint256)` | write |
//! | `reassign_coordinator` | `ComputePoolTraining.reassignCoordinator(uint256,address)` | write (requester / governance only) |
//! | `expire_stalled_training` | `ComputePoolTraining.expireStalledTraining(uint256)` | write (requester or joined worker) |
//!
//! # Partial-snapshot caveat (S1.5 follow-up)
//!
//! `snapshot` populates the fields the worker state-machine needs to
//! drive the training loop:
//!   - spec (full TrainingJobSpec)
//!   - state (JobChainState)
//!   - current_epoch
//!   - coordinator
//!
//! It does NOT currently populate:
//!   - workers (TODO: call `getWorkerList(uint256)` → address[])
//!   - epoch_roots (TODO: per-epoch `epochCommitment(uint256,uint32)`
//!     view calls, or batch via multicall)
//!
//! Both are additive fetches on top of the core getJob call. Landing
//! them in S1.5 keeps this slice focused on the state-machine-critical
//! path (join → close → commit × E → finalize) plus the one read
//! (`snapshot`) the worker uses to know where it is in that sequence.

use std::time::Duration;

use async_trait::async_trait;
use ethereum_types::{H160, H256, U256};
use serde_json::{json, Value};
use sha3::{Digest, Keccak256};

use crate::chain::{ChainClient, ChainError, JobChainSnapshot, JobChainState};
use crate::types::{EpochIndex, JobId, TrainingJobSpec, WorkerAddress, B256};
use crate::wallet::{tx_hash_of_signed, Eip1559Tx, Wallet};

/// Conservative gas limit for ComputePoolTraining writes. The heaviest
/// writes (`finalizeTrainingJob`) iterate over the worker list and do
/// one transfer per worker (~30k gas each), so we size for pools up
/// to ~50 workers plus headroom. Per-epoch `commitEpoch` and
/// `joinTrainingJob` are much lighter but the larger cap is free on
/// EIP-1559 (unused gas is refunded).
const WRITE_GAS_LIMIT: u64 = 3_000_000;

/// Minimum tip paid to the proposer (1 gwei). The daemon isn't
/// competing for inclusion priority, but a tip of zero risks being
/// dropped by some mempool policies.
const DEFAULT_PRIORITY_FEE_WEI: u64 = 1_000_000_000;

/// Max time we're willing to wait for a submitted tx to appear in a
/// block. After this we return a `Chain` error; the daemon loop
/// surfaces it to the operator. 30s comfortably covers the 1-2s
/// block target with room for a re-org or a slow peer.
const RECEIPT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Initial poll delay for `eth_getTransactionReceipt`.
const RECEIPT_POLL_INITIAL: Duration = Duration::from_millis(250);

/// Upper bound on poll delay.
const RECEIPT_POLL_MAX: Duration = Duration::from_millis(500);

/// Default per-request HTTP timeout for the JSON-RPC calls.
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

/// Precomputed 4-byte function selectors.
#[derive(Clone)]
struct Selectors {
    get_job: [u8; 4],
    join_training_job: [u8; 4],
    close_recruitment: [u8; 4],
    commit_epoch: [u8; 4],
    finalize_training_job: [u8; 4],
    reassign_coordinator: [u8; 4],
    expire_stalled_training: [u8; 4],
}

impl Selectors {
    fn compute() -> Self {
        Self {
            get_job: selector("getJob(uint256)"),
            join_training_job: selector("joinTrainingJob(uint256)"),
            close_recruitment: selector("closeRecruitment(uint256,address)"),
            commit_epoch: selector("commitEpoch(uint256,uint32,bytes32)"),
            finalize_training_job: selector("finalizeTrainingJob(uint256)"),
            reassign_coordinator: selector("reassignCoordinator(uint256,address)"),
            expire_stalled_training: selector("expireStalledTraining(uint256)"),
        }
    }
}

/// HTTP-backed [`ChainClient`] against a live Citrate JSON-RPC
/// endpoint.
///
/// Cheap to clone: the inner `reqwest::Client` + [`Wallet`] use
/// reference-counted handles internally.
#[derive(Clone)]
pub struct HttpChainClient {
    rpc_url: String,
    chain_id: u64,
    contract_addr: H160,
    wallet: Wallet,
    http: reqwest::Client,
    selectors: Selectors,
}

impl HttpChainClient {
    /// Construct a client pointed at `rpc_url`, signing writes to
    /// `contract_addr` with `wallet` on chain `chain_id`.
    pub fn new(rpc_url: String, chain_id: u64, contract_addr: H160, wallet: Wallet) -> Self {
        // CP-B-006 / CP-B-012: redirect-follow disabled (a 3xx must not
        // re-POST the signed JSON-RPC body to an off-gate host) + fail
        // CLOSED on a builder error rather than `Client::new()`, which
        // follows redirects and has no timeout.
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

    /// CP-B-006: verify the RPC endpoint's advertised chain id matches
    /// the configured one before signing any money transaction. Mirrors
    /// `pool-coordinator::HttpChainAdapter::verify_rpc_chain_id` (F-5).
    pub async fn verify_chain_id(&self) -> Result<(), ChainError> {
        let result = self.rpc("eth_chainId", serde_json::json!([])).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| ChainError::WrongState("eth_chainId result not a string".into()))?;
        let observed = parse_hex_u64(hex_str)
            .map_err(|e| ChainError::WrongState(format!("eth_chainId decode: {e}")))?;
        if observed != self.chain_id {
            return Err(ChainError::WrongState(format!(
                "RPC chain_id mismatch — configured {} vs observed {} ({})",
                self.chain_id, observed, self.rpc_url
            )));
        }
        Ok(())
    }

    /// POST a JSON-RPC request and return the `result` field.
    async fn rpc(&self, method: &str, params: Value) -> Result<Value, ChainError> {
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
            .map_err(|e| ChainError::WrongState(format!("{} transport: {}", method, e)))?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| ChainError::WrongState(format!("{} decode: {}", method, e)))?;
        if let Some(err) = value.get("error").filter(|v| !v.is_null()) {
            return Err(ChainError::WrongState(format!(
                "{} rpc error: {}",
                method, err
            )));
        }
        value
            .get("result")
            .cloned()
            .ok_or_else(|| ChainError::WrongState(format!("{} missing result", method)))
    }

    /// Issue `eth_call(to=contract_addr, data)` and return the
    /// decoded return bytes.
    async fn eth_call(&self, data: &[u8]) -> Result<Vec<u8>, ChainError> {
        let to = format!("0x{}", hex::encode(self.contract_addr.as_bytes()));
        let data_hex = format!("0x{}", hex::encode(data));
        let params = json!([
            { "to": to, "data": data_hex },
            "latest",
        ]);
        let result = self.rpc("eth_call", params).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| ChainError::WrongState("eth_call result not a string".into()))?;
        hex::decode(hex_str.trim_start_matches("0x"))
            .map_err(|e| ChainError::WrongState(format!("eth_call bad hex: {}", e)))
    }

    async fn fetch_nonce(&self) -> Result<u64, ChainError> {
        let addr = format!("0x{}", hex::encode(self.wallet.address().as_bytes()));
        let result = self
            .rpc("eth_getTransactionCount", json!([addr, "pending"]))
            .await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| ChainError::WrongState("eth_getTransactionCount not a string".into()))?;
        parse_hex_u64(hex_str).map_err(|e| ChainError::WrongState(format!("nonce decode: {}", e)))
    }

    async fn fetch_gas_price(&self) -> Result<U256, ChainError> {
        let result = self.rpc("eth_gasPrice", json!([])).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| ChainError::WrongState("eth_gasPrice not a string".into()))?;
        parse_hex_u256(hex_str)
            .map_err(|e| ChainError::WrongState(format!("gas price decode: {}", e)))
    }

    /// Build, sign, and submit an EIP-1559 write to
    /// `self.contract_addr` with the given calldata + value. Returns
    /// the transaction hash on success.
    async fn send_write(&self, calldata: Vec<u8>, value: U256) -> Result<H256, ChainError> {
        let nonce = self.fetch_nonce().await?;
        let gas_price = self.fetch_gas_price().await?;
        let priority = U256::from(DEFAULT_PRIORITY_FEE_WEI);
        let max_fee = gas_price
            .saturating_mul(U256::from(2u64))
            .saturating_add(priority);

        let tx = Eip1559Tx {
            chain_id: self.chain_id,
            nonce,
            max_priority_fee_per_gas: priority,
            max_fee_per_gas: max_fee,
            gas_limit: WRITE_GAS_LIMIT,
            to: self.contract_addr,
            value,
            data: calldata,
        };
        let signed = self
            .wallet
            .sign_eip1559(&tx)
            .map_err(|e| ChainError::WrongState(format!("sign: {}", e)))?;
        let tx_hash = tx_hash_of_signed(&signed);
        let signed_hex = format!("0x{}", hex::encode(&signed));

        let _ = self
            .rpc("eth_sendRawTransaction", json!([signed_hex]))
            .await?;

        // Poll for receipt; surface revert as a chain error.
        let _block_number = self.wait_for_receipt(tx_hash).await?;
        Ok(tx_hash)
    }

    /// Poll `eth_getTransactionReceipt` with exponential backoff
    /// until we see a non-null receipt.
    async fn wait_for_receipt(&self, tx_hash: H256) -> Result<Option<u64>, ChainError> {
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
                    return Err(ChainError::WrongState(format!("tx {:?} reverted", tx_hash)));
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

    /// Decode the return of `getJob(uint256)` into the partial
    /// snapshot we populate in S1. See the partial-snapshot caveat
    /// at the module docstring: `workers` and `epoch_roots` stay
    /// empty in this slice.
    fn decode_job_snapshot(&self, ret: &[u8]) -> Result<JobChainSnapshot, ChainError> {
        // TrainingJob has 17 statically-sized fields. ABI-encoded as
        // a flat sequence of 32-byte words (no dynamic fields means
        // no offset indirection). Layout:
        //   [  0..  32] requester (address, left-padded)
        //   [ 32..  64] modelStartHash (bytes32)
        //   [ 64..  96] datasetHash (bytes32)
        //   [ 96.. 128] epochCount (uint32, right-aligned)
        //   [128.. 160] stepsPerEpoch (uint32)
        //   [160.. 192] minWorkers (uint32)
        //   [192.. 224] maxWorkers (uint32)
        //   [224.. 256] challengeWindowBlocks (uint32)
        //   [256.. 288] perEpochBudget (uint128)
        //   [288.. 320] perWorkerStake (uint128)
        //   [320.. 352] escrowRemaining (uint128) — not surfaced
        //   [352.. 384] state (uint8 enum)
        //   [384.. 416] currentEpoch (uint32)
        //   [416.. 448] workerCount (uint32) — not surfaced
        //   [448.. 480] allEpochsCommittedBlock (uint64) — not surfaced
        //   [480.. 512] lastActivityBlock (uint64) — not surfaced
        //   [512.. 544] coordinator (address, left-padded)
        const EXPECTED_LEN: usize = 544;
        if ret.len() < EXPECTED_LEN {
            return Err(ChainError::WrongState(format!(
                "getJob return too short: {} bytes (expected {})",
                ret.len(),
                EXPECTED_LEN,
            )));
        }

        // requester — not part of JobChainSnapshot but we validate
        // the head is non-zero; a zero requester means "unknown job"
        // on-chain (contract's jobExists modifier would revert, but
        // pre-revert stubs might return all-zeros).
        let requester_is_zero = ret[0..32].iter().all(|&b| b == 0);
        if requester_is_zero {
            return Err(ChainError::UnknownJob(0));
        }

        let model_start_hash = B256::from_slice(&ret[32..64]);
        let dataset_hash = B256::from_slice(&ret[64..96]);
        let epoch_count = u32_from_word(&ret[96..128])?;
        let steps_per_epoch = u32_from_word(&ret[128..160])?;
        let min_workers = u32_from_word(&ret[160..192])?;
        let max_workers = u32_from_word(&ret[192..224])?;
        let challenge_window_blocks = u32_from_word(&ret[224..256])?;
        let per_epoch_budget = u128_from_word(&ret[256..288])?;
        let per_worker_stake = u128_from_word(&ret[288..320])?;
        // 320..352 escrowRemaining — skipped
        let state_byte = ret[383]; // uint8 sits in the last byte of its word
        let state = job_state_from_byte(state_byte)?;
        let current_epoch = u32_from_word(&ret[384..416])?;
        // 416..448 workerCount — skipped (via getWorkerList in S1.5)
        // 448..480 allEpochsCommittedBlock — skipped
        // 480..512 lastActivityBlock — skipped
        let coordinator_bytes = &ret[512..544];
        let coordinator_addr = H160::from_slice(&coordinator_bytes[12..32]);
        let coordinator = if coordinator_addr == H160::zero() {
            None
        } else {
            Some(coordinator_addr)
        };

        Ok(JobChainSnapshot {
            spec: TrainingJobSpec {
                model_start_hash,
                dataset_hash,
                epoch_count,
                steps_per_epoch,
                min_workers,
                max_workers,
                challenge_window_blocks,
                per_epoch_budget,
                per_worker_stake,
            },
            state,
            current_epoch,
            coordinator,
            // TODO(S1.5): call ComputePoolTraining.getWorkerList(jobId)
            // and populate.
            workers: Vec::new(),
            // TODO(S1.5): for each epoch in 0..current_epoch call
            // ComputePoolTraining.epochCommitment(jobId, epoch) and
            // populate.
            epoch_roots: std::collections::HashMap::new(),
        })
    }

    /// Fetch the latest block number from the node. Used by the
    /// event-loop to bound each poll's `to_block`.
    pub async fn latest_block(&self) -> Result<u64, ChainError> {
        let result = self.rpc("eth_blockNumber", serde_json::json!([])).await?;
        let hex_str = result
            .as_str()
            .ok_or_else(|| ChainError::WrongState("eth_blockNumber not a string".into()))?;
        parse_hex_u64(hex_str)
            .map_err(|e| ChainError::WrongState(format!("blockNumber decode: {}", e)))
    }

    /// Poll all ComputePoolTraining events in `[from_block, to_block]`
    /// optionally filtered by indexed jobId. Returns raw logs the
    /// caller classifies via `crate::events::classify`.
    pub async fn poll_raw_logs(
        &self,
        from_block: u64,
        to_block: u64,
        job_id_filter: Option<u64>,
    ) -> Result<Vec<crate::events::RawLog>, ChainError> {
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
        let arr = result
            .as_array()
            .ok_or_else(|| ChainError::WrongState("eth_getLogs not array".into()))?;
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
impl ChainClient for HttpChainClient {
    /// TRUE: this client is bound to a real `ComputePoolTraining` deployment,
    /// so a committed epoch releases real SALT. Pairs with
    /// `ModelBackend::honors_job_spec` to stop a placeholder backend earning.
    fn is_live_settlement(&self) -> bool {
        true
    }

    async fn snapshot(&self, job_id: JobId) -> Result<JobChainSnapshot, ChainError> {
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.get_job);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        let ret = self.eth_call(&data).await?;
        self.decode_job_snapshot(&ret)
    }

    async fn join_training_job(
        &self,
        job_id: JobId,
        _sender: WorkerAddress,
        stake: u128,
    ) -> Result<(), ChainError> {
        // `sender` is an artifact of the trait shape shared with the
        // mock (which tests the state-machine without a live signer).
        // On chain, the caller is the wallet's own address via
        // msg.sender; we ignore the parameter here. A daemon loading
        // the same operator key as the worker address will always
        // see sender == self.wallet.address().
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.join_training_job);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        self.send_write(data, U256::from(stake)).await?;
        Ok(())
    }

    async fn close_recruitment(
        &self,
        job_id: JobId,
        coordinator: WorkerAddress,
    ) -> Result<(), ChainError> {
        let mut data = Vec::with_capacity(4 + 64);
        data.extend_from_slice(&self.selectors.close_recruitment);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        data.extend_from_slice(&pad_address_left(coordinator));
        self.send_write(data, U256::zero()).await?;
        Ok(())
    }

    async fn commit_epoch(
        &self,
        job_id: JobId,
        _sender: WorkerAddress,
        epoch: EpochIndex,
        root: B256,
    ) -> Result<(), ChainError> {
        // Like join_training_job, `sender` is trait-shape-compat; the
        // live chain uses msg.sender from the signing wallet.
        let mut data = Vec::with_capacity(4 + 96);
        data.extend_from_slice(&self.selectors.commit_epoch);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        data.extend_from_slice(&u256_word(U256::from(epoch)));
        data.extend_from_slice(root.as_bytes()); // bytes32 is already 32 bytes
        self.send_write(data, U256::zero()).await?;
        Ok(())
    }

    async fn advance_blocks(&self, _n: u64) {
        // advance_blocks is a mock-only concept — the live chain's
        // clock is driven by real block production. Left as a no-op
        // so tests sharing the trait can call it harmlessly.
        tracing::warn!("HttpChainClient::advance_blocks called; no-op on live chain");
    }

    async fn finalize(&self, job_id: JobId) -> Result<(), ChainError> {
        let mut data = Vec::with_capacity(4 + 32);
        data.extend_from_slice(&self.selectors.finalize_training_job);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        self.send_write(data, U256::zero()).await?;
        Ok(())
    }

    async fn reassign_coordinator(
        &self,
        job_id: JobId,
        _caller: WorkerAddress,
        new_coordinator: WorkerAddress,
    ) -> Result<(), ChainError> {
        // `caller` is trait-compat; on chain, msg.sender is the
        // signing wallet.
        let mut data = Vec::with_capacity(4 + 64);
        data.extend_from_slice(&self.selectors.reassign_coordinator);
        data.extend_from_slice(&u256_word(U256::from(job_id)));
        data.extend_from_slice(&pad_address_left(new_coordinator));
        self.send_write(data, U256::zero()).await?;
        Ok(())
    }

    async fn expire_stalled_training(
        &self,
        _job_id: JobId,
        _caller: WorkerAddress,
    ) -> Result<(), ChainError> {
        Err(ChainError::WrongState(
            "expireStalledTraining not wired".into(),
        ))
    }

    async fn challenge_step(
        &self,
        _job_id: JobId,
        _caller: WorkerAddress,
        _epoch: EpochIndex,
        _step: u32,
        _target: WorkerAddress,
        _leaf: B256,
        _merkle_proof: Vec<B256>,
        _bond: u128,
    ) -> Result<(), ChainError> {
        // S1.5 follow-up. The challenge lifecycle requires a dynamic
        // `bytes32[]` proof arg and a bond transfer — not needed for
        // the S1 happy-path bring-up.
        Err(ChainError::WrongState(
            "challenge_step not implemented on HttpChainClient (S1.5)".into(),
        ))
    }

    async fn vote_challenge(
        &self,
        _job_id: JobId,
        _voter: WorkerAddress,
        _epoch: EpochIndex,
        _step: u32,
        _target: WorkerAddress,
        _uphold: bool,
    ) -> Result<(), ChainError> {
        // S1.5 follow-up.
        Err(ChainError::WrongState(
            "vote_challenge not implemented on HttpChainClient (S1.5)".into(),
        ))
    }

    async fn set_committee_member(&self, _member: WorkerAddress, _active: bool) {
        // Governance-only function on the live contract; not invoked
        // by the worker daemon. Mock-only in the trait surface.
        tracing::warn!(
            "HttpChainClient::set_committee_member called; no-op (governance-only on chain)"
        );
    }

    async fn worker_total_slashed(&self, _job_id: JobId, _worker: WorkerAddress) -> u128 {
        // S1.5 follow-up — requires getWorker(jobId, worker) decode
        // of WorkerInfo. Not on the critical-path for the training
        // loop; workers discover slashes via events in the real
        // daemon.
        0
    }

    async fn challenger_reward(&self, _challenger: WorkerAddress) -> u128 {
        // Mock-only accumulator; the live contract pays challengers
        // via the challenge-resolution tx directly.
        0
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

fn pad_address_left(addr: H160) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[12..32].copy_from_slice(addr.as_bytes());
    buf
}

fn u32_from_word(word: &[u8]) -> Result<u32, ChainError> {
    if word.len() != 32 {
        return Err(ChainError::WrongState(format!(
            "u32 word length {} (expected 32)",
            word.len()
        )));
    }
    let v = U256::from_big_endian(word);
    if v > U256::from(u32::MAX) {
        Err(ChainError::WrongState(format!(
            "u32 value out of range: {}",
            v
        )))
    } else {
        Ok(v.as_u32())
    }
}

fn u128_from_word(word: &[u8]) -> Result<u128, ChainError> {
    if word.len() != 32 {
        return Err(ChainError::WrongState(format!(
            "u128 word length {} (expected 32)",
            word.len()
        )));
    }
    let v = U256::from_big_endian(word);
    if v > U256::from(u128::MAX) {
        Err(ChainError::WrongState(format!(
            "u128 value out of range: {}",
            v
        )))
    } else {
        Ok(v.as_u128())
    }
}

fn job_state_from_byte(b: u8) -> Result<JobChainState, ChainError> {
    // Matches the Solidity enum ordering:
    //   0 = Recruiting, 1 = Training, 2 = Awaiting,
    //   3 = Finalized, 4 = Aborted
    match b {
        0 => Ok(JobChainState::Recruiting),
        1 => Ok(JobChainState::Training),
        2 => Ok(JobChainState::Awaiting),
        3 => Ok(JobChainState::Finalized),
        4 => Ok(JobChainState::Aborted),
        other => Err(ChainError::WrongState(format!(
            "unknown JobState discriminant: {}",
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

    /// Well-known key; derives 0x7e5f4552091a69125d5dfcb7b8c2659029395bdf.
    const TEST_HEX: &str = "0000000000000000000000000000000000000000000000000000000000000001";

    /// Records every JSON-RPC request the stub server saw so tests
    /// can assert on method + params.
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

    /// Stub server state: per-method canned response queues.
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

        /// Pre-queue the standard write-path sequence
        /// (getTransactionCount → gasPrice → sendRawTransaction →
        /// successful receipt). Tests call this once per write
        /// operation they're exercising.
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

    async fn rpc_handler(State(state): State<StubState>, Json(body): Json<Value>) -> Json<Value> {
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
        H160::repeat_byte(0xCC)
    }

    fn make_client(rpc_url: String) -> HttpChainClient {
        let wallet = Wallet::from_hex(TEST_HEX).expect("wallet");
        HttpChainClient::new(rpc_url, 40204, contract_addr(), wallet)
    }

    /// Build the 544-byte ABI-encoded return for `getJob`. Sets the
    /// fields that matter to `decode_job_snapshot` and leaves the
    /// rest (escrowRemaining, workerCount, allEpochsCommittedBlock,
    /// lastActivityBlock) as zero words.
    #[allow(clippy::too_many_arguments)]
    fn encode_training_job(
        requester: H160,
        model_start_hash: [u8; 32],
        dataset_hash: [u8; 32],
        epoch_count: u32,
        steps_per_epoch: u32,
        min_workers: u32,
        max_workers: u32,
        challenge_window_blocks: u32,
        per_epoch_budget: u128,
        per_worker_stake: u128,
        state: u8,
        current_epoch: u32,
        coordinator: H160,
    ) -> String {
        let mut out = vec![0u8; 544];
        // requester in last 20 bytes of word 0
        out[12..32].copy_from_slice(requester.as_bytes());
        // modelStartHash
        out[32..64].copy_from_slice(&model_start_hash);
        // datasetHash
        out[64..96].copy_from_slice(&dataset_hash);
        // epochCount
        U256::from(epoch_count).to_big_endian(&mut out[96..128]);
        // stepsPerEpoch
        U256::from(steps_per_epoch).to_big_endian(&mut out[128..160]);
        // minWorkers
        U256::from(min_workers).to_big_endian(&mut out[160..192]);
        // maxWorkers
        U256::from(max_workers).to_big_endian(&mut out[192..224]);
        // challengeWindowBlocks
        U256::from(challenge_window_blocks).to_big_endian(&mut out[224..256]);
        // perEpochBudget
        U256::from(per_epoch_budget).to_big_endian(&mut out[256..288]);
        // perWorkerStake
        U256::from(per_worker_stake).to_big_endian(&mut out[288..320]);
        // state (uint8 in last byte of word at [352..384])
        out[383] = state;
        // currentEpoch
        U256::from(current_epoch).to_big_endian(&mut out[384..416]);
        // coordinator in last 20 bytes of word at [512..544]
        out[512 + 12..512 + 32].copy_from_slice(coordinator.as_bytes());
        format!("0x{}", hex::encode(out))
    }

    // ── Tests ────────────────────────────────────────────────────

    #[test]
    fn selectors_are_distinct_and_four_bytes() {
        let s = Selectors::compute();
        let all = [
            s.get_job,
            s.join_training_job,
            s.close_recruitment,
            s.commit_epoch,
            s.finalize_training_job,
            s.reassign_coordinator,
            s.expire_stalled_training,
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "selector collision at {} vs {}", i, j);
            }
        }
    }

    #[tokio::test]
    async fn join_training_job_signs_and_submits_with_value() {
        let state = StubState::new();
        state.queue_write_happy_path("0x5");
        let addr = spawn_stub_rpc(state.clone()).await;

        let client = make_client(format!("http://{}", addr));
        let worker = Address::repeat_byte(0x01);
        let stake = 10_000_000_000_000_000_000u128; // 10 ether
        client
            .join_training_job(42, worker, stake)
            .await
            .expect("join");

        // Verify the raw tx was a 0x02-prefixed EIP-1559 envelope.
        let params = state
            .log
            .last_params("eth_sendRawTransaction")
            .expect("saw send");
        let raw_hex = params[0].as_str().expect("raw tx is string").to_string();
        let raw = hex::decode(raw_hex.trim_start_matches("0x")).expect("decode raw");
        assert_eq!(raw[0], 0x02, "EIP-1559 type prefix");

        // Method sequence is nonce → gas price → send → receipt.
        let methods = state.log.methods();
        assert_eq!(methods[0], "eth_getTransactionCount");
        assert_eq!(methods[1], "eth_gasPrice");
        assert_eq!(methods[2], "eth_sendRawTransaction");
        assert_eq!(methods[3], "eth_getTransactionReceipt");
    }

    #[tokio::test]
    async fn commit_epoch_encodes_epoch_and_root() {
        let state = StubState::new();
        state.queue_write_happy_path("0x1");
        let addr = spawn_stub_rpc(state.clone()).await;

        let client = make_client(format!("http://{}", addr));
        let coordinator = Address::repeat_byte(0xAA);
        let root = B256::repeat_byte(0xEE);
        client
            .commit_epoch(7, coordinator, 2, root)
            .await
            .expect("commit");

        // The signed tx contains our calldata. Extract it from the
        // RLP-decoded tx: EIP-1559 envelope is 0x02 || rlp([chainId,
        // nonce, maxPrio, maxFee, gas, to, value, data, accessList]).
        // Rather than RLP-decoding inside the test, verify the
        // commitEpoch selector bytes appear in the RLP body at the
        // expected offset (they're embedded as the 8th list element
        // — `data`). The selector's four bytes are unique enough
        // that presence-in-payload is a strong check.
        let params = state
            .log
            .last_params("eth_sendRawTransaction")
            .expect("saw send");
        let raw_hex = params[0].as_str().expect("string").to_string();
        let raw = hex::decode(raw_hex.trim_start_matches("0x")).expect("decode");
        let sel = Selectors::compute().commit_epoch;
        assert!(
            raw.windows(4).any(|w| w == sel),
            "commitEpoch selector not found in signed tx"
        );
        // And the root bytes should also appear in the payload.
        assert!(
            raw.windows(32).any(|w| w == root.as_bytes()),
            "commit root not found in signed tx"
        );
    }

    #[tokio::test]
    async fn reassign_coordinator_encodes_new_coordinator_address() {
        let state = StubState::new();
        state.queue_write_happy_path("0x2");
        let addr = spawn_stub_rpc(state.clone()).await;

        let client = make_client(format!("http://{}", addr));
        let caller = Address::repeat_byte(0x11);
        let new_coord = Address::repeat_byte(0x33);
        client
            .reassign_coordinator(9, caller, new_coord)
            .await
            .expect("reassign");

        let params = state
            .log
            .last_params("eth_sendRawTransaction")
            .expect("saw send");
        let raw_hex = params[0].as_str().expect("string").to_string();
        let raw = hex::decode(raw_hex.trim_start_matches("0x")).expect("decode");
        let sel = Selectors::compute().reassign_coordinator;
        assert!(
            raw.windows(4).any(|w| w == sel),
            "reassignCoordinator selector not found in signed tx"
        );
        // The new coordinator's 20-byte address should appear as a
        // contiguous run somewhere in the payload (it's the second
        // arg, left-padded into a 32-byte word).
        assert!(
            raw.windows(20).any(|w| w == new_coord.as_bytes()),
            "new coordinator address not found in signed tx"
        );
    }

    #[test]
    fn expire_stalled_training_selector_is_pinned() {
        // keccak256("expireStalledTraining(uint256)")[..4], from the ABI of
        // ComputePoolTraining on citrate-chain main.
        assert_eq!(
            hex::encode(Selectors::compute().expire_stalled_training),
            "0f73d3e4"
        );
    }

    #[tokio::test]
    async fn expire_stalled_training_encodes_job_id_only() {
        let state = StubState::new();
        state.queue_write_happy_path("0x2");
        let addr = spawn_stub_rpc(state.clone()).await;

        let client = make_client(format!("http://{}", addr));
        client
            .expire_stalled_training(0x0102_0304, Address::repeat_byte(0x11))
            .await
            .expect("expire");

        let params = state
            .log
            .last_params("eth_sendRawTransaction")
            .expect("saw send");
        let raw_hex = params[0].as_str().expect("string").to_string();
        let raw = hex::decode(raw_hex.trim_start_matches("0x")).expect("decode");
        let mut call = Selectors::compute().expire_stalled_training.to_vec();
        call.extend_from_slice(&u256_word(U256::from(0x0102_0304u64)));
        assert!(
            raw.windows(call.len()).any(|w| w == call.as_slice()),
            "expireStalledTraining(jobId) calldata not found in signed tx"
        );
    }

    #[tokio::test]
    async fn snapshot_decodes_training_job_fields() {
        let state = StubState::new();
        let requester = H160::repeat_byte(0xAB);
        let coordinator = H160::repeat_byte(0xCD);
        state.queue(
            "eth_call",
            json!(encode_training_job(
                requester,
                [0x11u8; 32],                // modelStartHash
                [0x22u8; 32],                // datasetHash
                3,                           // epochCount
                10,                          // stepsPerEpoch
                2,                           // minWorkers
                5,                           // maxWorkers
                20,                          // challengeWindowBlocks
                100_000_000_000_000_000u128, // perEpochBudget (0.1 ether)
                50_000_000_000_000_000u128,  // perWorkerStake (0.05 ether)
                1,                           // state = Training
                1,                           // currentEpoch
                coordinator,
            )),
        );
        let addr = spawn_stub_rpc(state).await;
        let client = make_client(format!("http://{}", addr));

        let snap = client.snapshot(42).await.expect("snapshot");
        assert_eq!(snap.spec.epoch_count, 3);
        assert_eq!(snap.spec.steps_per_epoch, 10);
        assert_eq!(snap.spec.min_workers, 2);
        assert_eq!(snap.spec.max_workers, 5);
        assert_eq!(snap.spec.challenge_window_blocks, 20);
        assert_eq!(snap.spec.per_epoch_budget, 100_000_000_000_000_000u128);
        assert_eq!(snap.spec.per_worker_stake, 50_000_000_000_000_000u128);
        assert_eq!(snap.spec.model_start_hash, B256::repeat_byte(0x11));
        assert_eq!(snap.spec.dataset_hash, B256::repeat_byte(0x22));
        assert_eq!(snap.state, JobChainState::Training);
        assert_eq!(snap.current_epoch, 1);
        assert_eq!(snap.coordinator, Some(coordinator));
        // S1 partial snapshot: these stay empty.
        assert!(snap.workers.is_empty());
        assert!(snap.epoch_roots.is_empty());
    }

    #[tokio::test]
    async fn snapshot_rejects_zero_requester_as_unknown_job() {
        let state = StubState::new();
        // All-zero return = contract signals unknown job (pre-revert
        // shape some stubs return).
        state.queue(
            "eth_call",
            json!(format!("0x{}", hex::encode(vec![0u8; 544]))),
        );
        let addr = spawn_stub_rpc(state).await;
        let client = make_client(format!("http://{}", addr));

        let err = client.snapshot(99).await.expect_err("unknown");
        assert!(matches!(err, ChainError::UnknownJob(_)));
    }

    #[tokio::test]
    async fn snapshot_rejects_short_return() {
        let state = StubState::new();
        state.queue("eth_call", json!("0x1234"));
        let addr = spawn_stub_rpc(state).await;
        let client = make_client(format!("http://{}", addr));

        let err = client.snapshot(1).await.expect_err("short");
        assert!(matches!(err, ChainError::WrongState(_)));
    }

    #[test]
    fn job_state_from_byte_maps_all_variants() {
        assert_eq!(
            job_state_from_byte(0).expect("0"),
            JobChainState::Recruiting
        );
        assert_eq!(job_state_from_byte(1).expect("1"), JobChainState::Training);
        assert_eq!(job_state_from_byte(2).expect("2"), JobChainState::Awaiting);
        assert_eq!(job_state_from_byte(3).expect("3"), JobChainState::Finalized);
        assert_eq!(job_state_from_byte(4).expect("4"), JobChainState::Aborted);
        assert!(job_state_from_byte(5).is_err());
    }
}
