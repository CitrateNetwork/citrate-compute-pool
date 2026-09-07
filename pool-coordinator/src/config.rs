//! Coordinator daemon configuration.

use std::collections::HashMap;
use std::env;

use ethereum_types::{H160, U256};

use crate::outbound::validate_outbound_url;

/// Runtime config. Operators populate this from env vars in `main.rs`.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Citrate chain id (40204 testnet).
    pub chain_id: u64,
    /// JSON-RPC endpoint for chain reads + tx submission.
    pub rpc_url: String,
    /// Address this daemon's wallet signs as. Must equal the
    /// secp256k1 keystore's address.
    pub wallet_address: H160,
    /// Map of pool member address → endpoint of the member's
    /// `/pool-infer` handler. Operator-supplied; mismatch produces
    /// `UnknownMemberEndpoint` at dispatch time. FWA-C8-01: each URL is
    /// enforced through the outbound TLS gate at parse time (https any
    /// host / http loopback only), so this is HTTPS in production.
    pub member_endpoints: HashMap<H160, String>,
    /// Per-request HTTPS timeout against pool members.
    pub provider_timeout_secs: u64,
    /// CP-B-008 admission floor: refuse a `ComputeRequested` whose
    /// escrowed `payment_grains` is below this, BEFORE `recordDispatch`.
    /// The dispatch decision never reads escrow otherwise, so a
    /// zero-payment job would be recorded on-chain against a
    /// deterministically-selected honest member and then `failJob`'d
    /// against it. Default 1 (refuse zero-payment). `0` disables.
    pub min_payment_grains: U256,
    /// CP-B-008 admission cap: refuse a job whose decoded prompt exceeds
    /// this many bytes before dispatch. Bounds the multi-megabyte-prompt
    /// grief vector. Default 128 KiB.
    pub max_prompt_bytes: usize,
    /// CP-B-008 admission cap: refuse a job whose `max_tokens` exceeds
    /// this before dispatch. The decoder clamps only at `u32::MAX`;
    /// this is the real serviceable ceiling. Default 8192.
    pub max_tokens_cap: u32,
}

impl CoordinatorConfig {
    /// Load from env. Returns `Err(String)` describing the missing
    /// variable on any failure — `main.rs` prints + exits 1.
    pub fn from_env() -> Result<Self, String> {
        let chain_id = env::var("CITRATE_POOL_CHAIN_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(40204);
        let rpc_url = env::var("CITRATE_POOL_RPC_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8545".to_string());
        // FWA-C8-01: the RPC leg reads chain truth (coordinator
        // election, pool membership, job spec) and carries signed
        // writes — a MITM on a plaintext remote RPC can feed false
        // chain-truth. Fail closed at config load. Default is loopback,
        // so production deployments using the default are unaffected.
        validate_outbound_url(&rpc_url).map_err(|e| format!("CITRATE_POOL_RPC_URL: {}", e))?;
        let wallet_hex = env::var("CITRATE_POOL_WALLET_ADDRESS")
            .map_err(|_| "CITRATE_POOL_WALLET_ADDRESS unset".to_string())?;
        let wallet_address =
            parse_addr(&wallet_hex).map_err(|e| format!("CITRATE_POOL_WALLET_ADDRESS: {}", e))?;
        let endpoints_raw = env::var("CITRATE_POOL_MEMBER_ENDPOINTS").unwrap_or_default();
        let member_endpoints = parse_endpoints(&endpoints_raw)?;
        let provider_timeout_secs = env::var("CITRATE_POOL_PROVIDER_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        // CP-B-008 admission gate thresholds (env-overridable).
        let min_payment_grains = env::var("CITRATE_POOL_MIN_PAYMENT_GRAINS")
            .ok()
            .and_then(|s| U256::from_dec_str(s.trim()).ok())
            .unwrap_or_else(|| U256::from(1u64));
        let max_prompt_bytes = env::var("CITRATE_POOL_MAX_PROMPT_BYTES")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(128 * 1024);
        let max_tokens_cap = env::var("CITRATE_POOL_MAX_TOKENS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(8192);
        Ok(Self {
            chain_id,
            rpc_url,
            wallet_address,
            member_endpoints,
            provider_timeout_secs,
            min_payment_grains,
            max_prompt_bytes,
            max_tokens_cap,
        })
    }
}

fn parse_addr(s: &str) -> Result<H160, String> {
    let stripped = s.trim().strip_prefix("0x").unwrap_or(s.trim());
    if stripped.len() != 40 {
        return Err(format!("expected 40 hex chars, got {}", stripped.len()));
    }
    let bytes = hex::decode(stripped).map_err(|e| e.to_string())?;
    let mut a = [0u8; 20];
    a.copy_from_slice(&bytes);
    Ok(H160::from(a))
}

/// Parse `addr1=url1,addr2=url2,...` into a map.
fn parse_endpoints(s: &str) -> Result<HashMap<H160, String>, String> {
    let mut out = HashMap::new();
    for entry in s.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (addr, url) = entry
            .split_once('=')
            .ok_or_else(|| format!("bad entry (need addr=url): {}", entry))?;
        let parsed = parse_addr(addr).map_err(|e| format!("{}: {}", addr, e))?;
        let url = url.trim();
        // FWA-C8-01: the dispatch leg POSTs the buyer prompt to this
        // URL and trusts the completion to decide completeJob/failJob.
        // A remote-plaintext endpoint is MITM-able (read prompt, forge
        // completion, or forge failure). Refuse fail-closed at parse
        // time so a misconfigured operator dies at startup, not mid-job.
        // Mirrors node-agent's chainio::outbound gate (FUA-NODE-AGENT-06).
        validate_outbound_url(url).map_err(|e| format!("member endpoint {}: {}", addr, e))?;
        out.insert(parsed, url.to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_addr_accepts_with_or_without_0x() {
        let with = parse_addr("0x1111111111111111111111111111111111111111").expect("with");
        let without = parse_addr("1111111111111111111111111111111111111111").expect("without");
        assert_eq!(with, without);
    }

    #[test]
    fn parse_addr_rejects_short() {
        assert!(parse_addr("0xabc").is_err());
    }

    #[test]
    fn parse_endpoints_parses_two_entries() {
        // FWA-C8-01: endpoints now run through the outbound TLS gate at
        // parse time, so fixtures use the legitimate https shape.
        let s = "0xa1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1=https://m1.pool.example/infer,\
                 0xb2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2=https://m2.pool.example/infer";
        let map = parse_endpoints(s).expect("ok");
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn parse_endpoints_skips_blanks() {
        let map = parse_endpoints("").expect("ok");
        assert!(map.is_empty());
        let map2 = parse_endpoints(",,,").expect("ok");
        assert!(map2.is_empty());
    }

    #[test]
    fn parse_endpoints_rejects_missing_equals() {
        assert!(parse_endpoints("0x1111111111111111111111111111111111111111").is_err());
    }

    // FWA-C8-01 RED→tripwire (federation-wide audit 2026-06-20, MEDIUM):
    // the coordinator must NOT accept a remote-plaintext member endpoint —
    // a network MITM on the plaintext leg can read the buyer prompt and
    // forge a completion (pay-for-fabricated-work) or a failure (grief an
    // honest member). This mirrors node-agent's chainio::outbound gate
    // (FUA-NODE-AGENT-06). Pre-fix this PASSED (endpoint stored verbatim);
    // post-fix parse_endpoints runs every value through validate_outbound_url.
    #[test]
    fn red_fwa_c8_01_remote_plaintext_member_endpoint_is_refused() {
        // Serialise against the env-override test (shared process env).
        let _g = crate::outbound::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let raw = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=http://attacker.example/infer";
        assert!(
            parse_endpoints(raw).is_err(),
            "remote plaintext http:// member endpoint must be refused (MITM-able)"
        );
    }

    // The gate must still permit the legitimate shapes: https to any host,
    // and plaintext http only to loopback (local provider on the same box).
    #[test]
    fn parse_endpoints_accepts_https_and_loopback_http() {
        let _g = crate::outbound::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let https = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=https://m1.pool.example/infer";
        assert!(
            parse_endpoints(https).is_ok(),
            "https endpoint must be accepted"
        );
        let loop_http = "0xb2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2=http://127.0.0.1:8080/infer";
        assert!(
            parse_endpoints(loop_http).is_ok(),
            "loopback http must be accepted"
        );
    }
}
