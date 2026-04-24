//! Coordinator daemon configuration.

use std::collections::HashMap;
use std::env;

use ethereum_types::H160;

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
    /// Map of pool member address → HTTPS endpoint of the member's
    /// `/pool-infer` handler. Operator-supplied; mismatch produces
    /// `UnknownMemberEndpoint` at dispatch time.
    pub member_endpoints: HashMap<H160, String>,
    /// Per-request HTTPS timeout against pool members.
    pub provider_timeout_secs: u64,
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
            .unwrap_or_else(|_| "http://127.0.0.1:18545".to_string());
        let wallet_hex = env::var("CITRATE_POOL_WALLET_ADDRESS")
            .map_err(|_| "CITRATE_POOL_WALLET_ADDRESS unset".to_string())?;
        let wallet_address = parse_addr(&wallet_hex)
            .map_err(|e| format!("CITRATE_POOL_WALLET_ADDRESS: {}", e))?;
        let endpoints_raw = env::var("CITRATE_POOL_MEMBER_ENDPOINTS")
            .unwrap_or_default();
        let member_endpoints = parse_endpoints(&endpoints_raw)?;
        let provider_timeout_secs = env::var("CITRATE_POOL_PROVIDER_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        Ok(Self {
            chain_id,
            rpc_url,
            wallet_address,
            member_endpoints,
            provider_timeout_secs,
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
        out.insert(parsed, url.trim().to_string());
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
        let s = "0xa1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1=http://m1/infer,\
                 0xb2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2=http://m2/infer";
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
}
