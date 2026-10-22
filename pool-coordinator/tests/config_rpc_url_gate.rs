//! `CoordinatorConfig::from_env` refuses a plaintext RPC URL to a remote host at
//! load time, before any other variable is read.
//!
//! This file holds ONE test on purpose: it sets process environment variables,
//! and an integration-test file is its own process, so no other test can race
//! on them. Nothing is dialled; the config loader only parses strings.

use citrate_pool_coordinator::CoordinatorConfig;

/// The error `from_env` stops with; panics with `why` if it loads.
fn refusal(why: &str) -> String {
    match CoordinatorConfig::from_env() {
        Ok(_) => panic!("{why}"),
        Err(e) => e,
    }
}

#[test]
fn from_env_refuses_a_plaintext_remote_rpc_url_and_accepts_tls() {
    std::env::remove_var("CITRATE_POOL_ALLOW_INSECURE_OUTBOUND");
    // Unset, so the first variable checked after the RPC URL fails with its own
    // name: that is how the test tells which check refused.
    std::env::remove_var("CITRATE_POOL_WALLET_ADDRESS");

    // TEST-NET-3, never routed.
    std::env::set_var("CITRATE_POOL_RPC_URL", "http://203.0.113.7:8545");
    let err = refusal("a plaintext remote RPC URL must be refused at config load");
    assert!(
        err.starts_with("CITRATE_POOL_RPC_URL") && err.contains("plaintext"),
        "the refusal must name the RPC URL, got: {err}"
    );

    // Control: TLS to any host passes the gate, so loading proceeds to the
    // next variable.
    std::env::set_var("CITRATE_POOL_RPC_URL", "https://rpc.example.invalid");
    let err = refusal("the wallet address is unset, so loading still stops");
    assert!(
        err.starts_with("CITRATE_POOL_WALLET_ADDRESS"),
        "an https RPC URL must pass the gate, got: {err}"
    );

    // Control: plaintext to loopback is allowed (local node).
    std::env::set_var("CITRATE_POOL_RPC_URL", "http://127.0.0.1:8545");
    let err = refusal("the wallet address is unset, so loading still stops");
    assert!(
        err.starts_with("CITRATE_POOL_WALLET_ADDRESS"),
        "a loopback RPC URL must pass the gate, got: {err}"
    );
}
