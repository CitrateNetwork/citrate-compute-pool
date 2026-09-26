//! Live compatibility check of the worker's stalled-coordinator exit
//! against the current `ComputePoolTraining` contract.
//!
//! Gated: runs only when `CITRATE_R2_ANVIL_ADDRS` (JSON with a
//! `ComputePoolTraining` address) is set; `CITRATE_R2_ANVIL_RPC` defaults
//! to `http://127.0.0.1:8599`. The node must be an anvil (unlocked default
//! accounts, `anvil_mine`). Default key #9 is the worker (signs through the
//! real `HttpChainClient`); unlocked account #0 is the requester.
//!
//! Asserts, on the deployed contract:
//!   - a joined worker's `reassignCoordinator` reverts;
//!   - `expireStalledTraining` reverts until STALL_EXPIRY_BLOCKS have
//!     passed, then moves the job to Awaiting.

use citrate_training_worker::chain::{JobChainState, STALL_EXPIRY_BLOCKS};
use citrate_training_worker::{ChainClient, HttpChainClient, Wallet};
use ethereum_types::{H160, U256};
use serde_json::{json, Value};
use sha3::{Digest, Keccak256};

const WORKER_KEY: &str = "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6";
const WORKER: &str = "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720";
const REQUESTER: &str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";

async fn rpc(url: &str, method: &str, params: Value) -> Value {
    let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
    let resp: Value = reqwest::Client::new()
        .post(url)
        .json(&body)
        .send()
        .await
        .expect("rpc send")
        .json()
        .await
        .expect("rpc json");
    assert!(resp.get("error").is_none(), "{method}: {resp}");
    resp["result"].clone()
}

fn sel(sig: &str) -> Vec<u8> {
    Keccak256::digest(sig.as_bytes())[..4].to_vec()
}

fn word(v: U256) -> [u8; 32] {
    let mut w = [0u8; 32];
    v.to_big_endian(&mut w);
    w
}

async fn requester_send(url: &str, to: H160, data: Vec<u8>, value: U256) {
    let hash = rpc(
        url,
        "eth_sendTransaction",
        json!([{
            "from": REQUESTER,
            "to": format!("{to:?}"),
            "data": format!("0x{}", hex::encode(data)),
            "value": format!("0x{value:x}"),
        }]),
    )
    .await;
    for _ in 0..100 {
        let r = rpc(url, "eth_getTransactionReceipt", json!([hash])).await;
        if let Some(s) = r.get("status").and_then(Value::as_str) {
            assert_eq!(s, "0x1", "requester tx reverted");
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("no receipt for requester tx");
}

async fn job_count(url: &str, training: H160) -> U256 {
    let out = rpc(
        url,
        "eth_call",
        json!([{"to": format!("{training:?}"), "data": format!("0x{}", hex::encode(sel("nextJobId()")))}, "latest"]),
    )
    .await;
    U256::from_str_radix(out.as_str().expect("hex").trim_start_matches("0x"), 16).expect("u256")
}

#[tokio::test]
async fn worker_stall_exit_matches_the_deployed_contract() {
    let Ok(path) = std::env::var("CITRATE_R2_ANVIL_ADDRS") else {
        eprintln!("skipping: CITRATE_R2_ANVIL_ADDRS not set");
        return;
    };
    let url =
        std::env::var("CITRATE_R2_ANVIL_RPC").unwrap_or_else(|_| "http://127.0.0.1:8599".into());
    let book: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("addrs")).expect("json");
    let training: H160 = book["ComputePoolTraining"]
        .as_str()
        .expect("addr")
        .parse()
        .expect("h160");
    let worker_addr: H160 = WORKER.parse().expect("worker");

    let stake = U256::from(10u64).pow(U256::from(18u64));
    let budget = stake;
    // requestTrainingJob((bytes32,bytes32,uint32,uint32,uint32,uint32,uint32,uint128,uint128))
    let job_id = job_count(&url, training).await;
    let mut data = sel(
        "requestTrainingJob((bytes32,bytes32,uint32,uint32,uint32,uint32,uint32,uint128,uint128))",
    );
    data.extend_from_slice(&[0x11; 32]);
    data.extend_from_slice(&[0x22; 32]);
    for v in [1u64, 1, 1, 1, 10] {
        data.extend_from_slice(&word(U256::from(v))); // epochs, steps, min, max, window
    }
    data.extend_from_slice(&word(budget));
    data.extend_from_slice(&word(stake));
    requester_send(&url, training, data, budget).await;

    let worker = HttpChainClient::new(
        url.clone(),
        40204,
        training,
        Wallet::from_hex(WORKER_KEY).expect("wallet"),
    );
    let job = job_id.as_u64();
    worker
        .join_training_job(job, worker_addr, stake.as_u128())
        .await
        .expect("worker joins");

    let mut close = sel("closeRecruitment(uint256,address)");
    close.extend_from_slice(&word(job_id));
    let mut coord = [0u8; 32];
    coord[12..].copy_from_slice(worker_addr.as_bytes());
    close.extend_from_slice(&coord);
    requester_send(&url, training, close, U256::zero()).await;
    assert_eq!(
        worker.snapshot(job).await.expect("snap").state,
        JobChainState::Training
    );

    // A joined worker cannot appoint the coordinator, even past the timeout.
    rpc(&url, "anvil_mine", json!(["0x65"])).await; // 101 blocks
    assert!(
        worker
            .reassign_coordinator(job, worker_addr, worker_addr)
            .await
            .is_err(),
        "worker reassignCoordinator must revert on the deployed contract"
    );

    // Not yet stalled.
    assert!(worker
        .expire_stalled_training(job, worker_addr)
        .await
        .is_err());
    assert_eq!(
        worker.snapshot(job).await.expect("snap").state,
        JobChainState::Training
    );

    // Past STALL_EXPIRY_BLOCKS the worker's exit succeeds.
    rpc(
        &url,
        "anvil_mine",
        json!([format!("0x{:x}", STALL_EXPIRY_BLOCKS + 1)]),
    )
    .await;
    worker
        .expire_stalled_training(job, worker_addr)
        .await
        .expect("expireStalledTraining after stall expiry");
    assert_eq!(
        worker.snapshot(job).await.expect("snap").state,
        JobChainState::Awaiting
    );
}
