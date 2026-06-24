//! `outbound` — TLS enforcement for the coordinator's outbound HTTP
//! endpoints (FWA-C8-01, federation-wide audit 2026-06-20).
//!
//! The coordinator makes two kinds of outbound request:
//!
//! - the **member dispatch** leg — POSTs the buyer prompt to a pool
//!   member's `/pool-infer` handler at a config-supplied URL
//!   (`CITRATE_POOL_MEMBER_ENDPOINTS`), then trusts the completion to
//!   decide `completeJob` vs `failJob`; and
//! - the **JSON-RPC** leg — reads chain truth (coordinator election,
//!   pool membership, job spec) and submits signed writes
//!   (`CITRATE_POOL_RPC_URL`).
//!
//! Pre-fix both used their URLs verbatim, so a plaintext `http://`
//! endpoint on a remote host was silently accepted and a network MITM
//! could read the prompt and forge a completion (pay-for-fabricated-
//! work), forge a failure (grief an honest member), or feed false
//! chain-truth over the RPC leg. This is the exact
//! FUA-NODE-AGENT-06 scenario the node-agent already fixed in
//! `chainio::outbound::validate_outbound_url`; this module mirrors that
//! gate one-for-one (separate workspace, so ported rather than shared).
//!
//! The rule:
//!
//! - `https://` — accepted for any host.
//! - `http://`  — accepted **only** for loopback hosts (`localhost`,
//!   `127.0.0.0/8`, `[::1]`). A co-located provider / local node stays
//!   plain.
//! - anything else (other schemes, missing scheme, empty/garbled host,
//!   userinfo smuggling) — refused, fail closed, at config-load time so
//!   a misconfigured daemon dies at startup, not mid-job.
//!
//! Dev-only escape hatch: `CITRATE_POOL_ALLOW_INSECURE_OUTBOUND=1`
//! permits plaintext to non-loopback hosts for LAN test rigs. It is
//! logged loudly on every use and must never be set in production — a
//! MITM between the coordinator and a member/RPC is exactly the
//! FWA-C8-01 scenario.

/// Env var gating the dev-only plaintext-to-non-loopback escape hatch.
/// Documented dev-only; never set this in production.
pub const ALLOW_INSECURE_ENV: &str = "CITRATE_POOL_ALLOW_INSECURE_OUTBOUND";

/// Why an outbound endpoint URL was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutboundUrlError {
    /// Scheme is neither `http` nor `https` (or missing entirely).
    UnsupportedScheme(String),
    /// Plain `http://` to a non-loopback host without the explicit
    /// dev-only `CITRATE_POOL_ALLOW_INSECURE_OUTBOUND=1` override.
    PlaintextNonLoopback(String),
    /// The URL's authority could not be parsed safely (empty host, userinfo).
    Malformed(String),
}

impl core::fmt::Display for OutboundUrlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OutboundUrlError::UnsupportedScheme(u) => write!(
                f,
                "outbound endpoint {u:?} has an unsupported scheme: only https:// (any host) or http:// (loopback) are allowed (FWA-C8-01)"
            ),
            OutboundUrlError::PlaintextNonLoopback(u) => write!(
                f,
                "outbound endpoint {u:?} is plaintext http:// to a non-loopback host; use https://, or set {ALLOW_INSECURE_ENV}=1 (dev-only) to override (FWA-C8-01)"
            ),
            OutboundUrlError::Malformed(u) => write!(
                f,
                "outbound endpoint {u:?} has an unparseable or unsafe authority (empty host / userinfo); refusing fail-closed (FWA-C8-01)"
            ),
        }
    }
}

impl std::error::Error for OutboundUrlError {}

/// Validate `url` as an outbound endpoint, reading the
/// [`ALLOW_INSECURE_ENV`] escape hatch from the environment. This is
/// what config-load calls on every member endpoint + the RPC URL.
pub fn validate_outbound_url(url: &str) -> Result<(), OutboundUrlError> {
    let allow_insecure = std::env::var(ALLOW_INSECURE_ENV).is_ok_and(|v| v.trim() == "1");
    validate_outbound_url_with(url, allow_insecure)
}

/// Pure core of [`validate_outbound_url`]: `allow_insecure` is the
/// explicit dev-only override (separated for deterministic, env-free
/// unit tests).
pub fn validate_outbound_url_with(
    url: &str,
    allow_insecure: bool,
) -> Result<(), OutboundUrlError> {
    let (scheme, rest) = match url.split_once("://") {
        Some((s, r)) => (s.to_ascii_lowercase(), r),
        None => return Err(OutboundUrlError::UnsupportedScheme(url.to_string())),
    };
    match scheme.as_str() {
        // TLS: acceptable to any host (still require a parseable authority).
        "https" => {
            host_of(rest).ok_or_else(|| OutboundUrlError::Malformed(url.to_string()))?;
            Ok(())
        }
        "http" => {
            let host = host_of(rest).ok_or_else(|| OutboundUrlError::Malformed(url.to_string()))?;
            if is_loopback_host(&host) {
                Ok(())
            } else if allow_insecure {
                eprintln!(
                    "SECURITY [FWA-C8-01]: {ALLOW_INSECURE_ENV}=1 — allowing PLAINTEXT outbound to non-loopback {url:?}; dev-only, never use in production"
                );
                Ok(())
            } else {
                Err(OutboundUrlError::PlaintextNonLoopback(url.to_string()))
            }
        }
        _ => Err(OutboundUrlError::UnsupportedScheme(url.to_string())),
    }
}

/// WebSocket sibling of [`validate_outbound_url`] for the opt-in
/// `CITRATE_POOL_WS_URL` event-subscription leg
/// ([`crate::ws_chain::WsChainSubscriber`]). A plaintext `ws://` remote
/// subscription feeds `ComputeRequested` events into the dispatch loop,
/// so it is the same MITM-able chain-truth vector as the RPC leg
/// (FWA-C8-01). Rule mirrors the http gate:
///
/// - `wss://` — accepted for any host.
/// - `ws://`  — accepted **only** for loopback hosts.
/// - anything else — refused, fail closed.
pub fn validate_outbound_ws_url(url: &str) -> Result<(), OutboundUrlError> {
    let allow_insecure = std::env::var(ALLOW_INSECURE_ENV).is_ok_and(|v| v.trim() == "1");
    validate_outbound_ws_url_with(url, allow_insecure)
}

/// Pure core of [`validate_outbound_ws_url`] (env-free for tests).
pub fn validate_outbound_ws_url_with(
    url: &str,
    allow_insecure: bool,
) -> Result<(), OutboundUrlError> {
    let (scheme, rest) = match url.split_once("://") {
        Some((s, r)) => (s.to_ascii_lowercase(), r),
        None => return Err(OutboundUrlError::UnsupportedScheme(url.to_string())),
    };
    match scheme.as_str() {
        "wss" => {
            host_of(rest).ok_or_else(|| OutboundUrlError::Malformed(url.to_string()))?;
            Ok(())
        }
        "ws" => {
            let host = host_of(rest).ok_or_else(|| OutboundUrlError::Malformed(url.to_string()))?;
            if is_loopback_host(&host) {
                Ok(())
            } else if allow_insecure {
                eprintln!(
                    "SECURITY [FWA-C8-01]: {ALLOW_INSECURE_ENV}=1 — allowing PLAINTEXT ws:// outbound to non-loopback {url:?}; dev-only, never use in production"
                );
                Ok(())
            } else {
                Err(OutboundUrlError::PlaintextNonLoopback(url.to_string()))
            }
        }
        _ => Err(OutboundUrlError::UnsupportedScheme(url.to_string())),
    }
}

/// Extract the host from the part after `scheme://`: strip path/query/
/// fragment, reject userinfo (`@`) and empty hosts (fail closed —
/// `None`), strip the port. IPv6 literals keep their brackets stripped
/// (`[::1]:80` → `::1`).
fn host_of(rest: &str) -> Option<String> {
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return None; // empty host or userinfo smuggling → fail closed
    }
    if let Some(v6) = authority.strip_prefix('[') {
        // `[::1]` or `[::1]:8545` — the literal is inside the brackets.
        let (inside, after) = v6.split_once(']')?;
        if !(after.is_empty() || after.starts_with(':')) {
            return None;
        }
        return Some(inside.to_string());
    }
    // Non-bracketed: at most one `:` (host:port); more is a malformed
    // (unbracketed-IPv6) authority → fail closed.
    let mut parts = authority.split(':');
    let host = parts.next()?.to_string();
    match (parts.next(), parts.next()) {
        (_, Some(_)) => None,                                        // host:p:q → malformed
        (Some(port), None) if port.parse::<u16>().is_err() => None, // non-numeric port
        _ if host.is_empty() => None,
        _ => Some(host),
    }
}

/// Is `host` (already stripped of brackets/port) a loopback destination?
/// Only the literal `localhost` and loopback IP literals qualify — any
/// other hostname would need DNS to prove loopback-ness, so it fails
/// closed.
fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(v4) = host.parse::<std::net::Ipv4Addr>() {
        return v4.is_loopback();
    }
    if let Ok(v6) = host.parse::<std::net::Ipv6Addr>() {
        return v6.is_loopback();
    }
    false
}

/// Process-wide lock serialising any test that mutates the
/// [`ALLOW_INSECURE_ENV`] var against tests that read it (env is
/// process-global; without this the override could leak into a
/// concurrently-running config/RPC validation test).
#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    // FWA-C8-01: plaintext http:// to a non-loopback host must be
    // refused; loopback may stay http; https is always acceptable.
    #[test]
    fn https_is_accepted_for_any_host() {
        assert!(validate_outbound_url_with("https://m1.pool.network", false).is_ok());
        assert!(validate_outbound_url_with("https://203.0.113.7:8545/path", false).is_ok());
    }

    #[test]
    fn plaintext_loopback_is_accepted() {
        for url in [
            "http://127.0.0.1:8545",
            "http://127.5.5.5:8080/ipfs",
            "http://localhost:8080",
            "http://LOCALHOST:8080",
            "http://[::1]:8545",
        ] {
            assert!(validate_outbound_url_with(url, false).is_ok(), "{url} rejected");
        }
    }

    #[test]
    fn plaintext_non_loopback_is_refused() {
        for url in [
            "http://203.0.113.7:8545",
            "http://attacker.example/infer",
            "http://rpc.citrate.network",
            "http://localhost.evil.example:8080", // not the literal localhost
            "http://[2001:db8::1]:8545",
            "http://192.168.1.50:8080/ipfs",
        ] {
            assert!(
                matches!(
                    validate_outbound_url_with(url, false),
                    Err(OutboundUrlError::PlaintextNonLoopback(_))
                ),
                "{url} was not refused as plaintext non-loopback"
            );
        }
    }

    #[test]
    fn malformed_or_unsupported_urls_fail_closed() {
        for url in [
            "",
            "ftp://203.0.113.7/weights",
            "file:///etc/passwd",
            "203.0.113.7:8545",        // no scheme
            "http://",                 // empty host
            "http://user@127.0.0.1:1", // userinfo smuggling → fail closed
        ] {
            assert!(validate_outbound_url_with(url, false).is_err(), "{url} accepted");
        }
    }

    // The documented dev-only escape hatch unlocks plaintext non-loopback.
    #[test]
    fn explicit_insecure_override_allows_plaintext_remote() {
        assert!(validate_outbound_url_with("http://192.168.1.50:8545", true).is_ok());
        // …but garbage is still garbage even with the override.
        assert!(validate_outbound_url_with("ftp://192.168.1.50", true).is_err());
    }

    // FWA-C8-01 (ws leg): wss any-host, ws loopback-only, else refused.
    #[test]
    fn ws_gate_matches_http_policy() {
        assert!(validate_outbound_ws_url_with("wss://node.citrate.network", false).is_ok());
        assert!(validate_outbound_ws_url_with("ws://127.0.0.1:18546", false).is_ok());
        assert!(validate_outbound_ws_url_with("ws://[::1]:18546", false).is_ok());
        assert!(matches!(
            validate_outbound_ws_url_with("ws://203.0.113.7:18546", false),
            Err(OutboundUrlError::PlaintextNonLoopback(_))
        ));
        // http/https schemes are not valid ws endpoints → refused.
        assert!(validate_outbound_ws_url_with("https://node.citrate.network", false).is_err());
        assert!(validate_outbound_ws_url_with("", false).is_err());
        // dev override unlocks plaintext remote ws.
        assert!(validate_outbound_ws_url_with("ws://203.0.113.7:18546", true).is_ok());
    }

    #[test]
    fn host_extraction_handles_ports_paths_and_brackets() {
        assert_eq!(host_of("127.0.0.1:8545/x?y#z").as_deref(), Some("127.0.0.1"));
        assert_eq!(host_of("[::1]:8545/ipfs").as_deref(), Some("::1"));
        assert_eq!(host_of("[::1]").as_deref(), Some("::1"));
        assert_eq!(host_of("host:notaport"), None);
        assert_eq!(host_of("::1:8545"), None); // unbracketed v6 → fail closed
        assert_eq!(host_of(""), None);
    }

    // Mutation-kill: the empty-host guard (`host.is_empty()` match arm)
    // must reject an authority whose host part is empty but whose
    // authority string is non-empty (`:8545`). Distinct from the
    // empty-authority guard so flipping it to `false` is observable.
    #[test]
    fn host_extraction_rejects_empty_host_with_port() {
        assert_eq!(host_of(":8545"), None);
        assert_eq!(host_of(":"), None);
    }

    // Mutation-kill: the userinfo guard is `is_empty() || contains('@')`.
    // Flipping `||`→`&&` would let `user@host` through (non-empty, has
    // `@`). A non-empty authority WITH `@` must still be refused.
    #[test]
    fn host_extraction_rejects_userinfo_smuggling() {
        assert_eq!(host_of("user@evil.example"), None);
        assert_eq!(host_of("user@127.0.0.1:8545"), None);
    }

    // Mutation-kill: the multi-colon `(_, Some(_))` match arm rejects an
    // unbracketed IPv6-shaped authority. With a NON-empty host part so
    // the later `host.is_empty()` arm can't mask the deletion.
    #[test]
    fn host_extraction_rejects_unbracketed_multicolon() {
        assert_eq!(host_of("h:1:2"), None);
        assert_eq!(host_of("a:b:c:d"), None);
    }

    // Mutation-kill: every OutboundUrlError variant's Display must carry
    // the FWA-C8-01 tag + a non-empty, variant-specific message (so
    // replacing fmt with Ok(default) — an empty string — is caught).
    #[test]
    fn error_display_messages_are_specific() {
        let scheme = OutboundUrlError::UnsupportedScheme("ftp://x".into()).to_string();
        assert!(scheme.contains("FWA-C8-01") && scheme.contains("unsupported scheme"));
        let plain = OutboundUrlError::PlaintextNonLoopback("http://x".into()).to_string();
        assert!(plain.contains("FWA-C8-01") && plain.contains("plaintext"));
        let malformed = OutboundUrlError::Malformed("http://".into()).to_string();
        assert!(malformed.contains("FWA-C8-01") && malformed.contains("authority"));
    }

    // Mutation-kill: the env-reading wrappers honour the dev-only
    // override ONLY when the var is exactly "1" (the `== "1"` check).
    // Serialised because env is process-global. Covers both http + ws
    // wrappers so the `validate_outbound_ws_url -> Ok(())` body mutant
    // and the `== / !=` guard mutants are observable through the public
    // env-reading entry points.
    #[test]
    fn env_wrappers_honour_override_exactly() {
        // Serialise against every other env-reading validation test so
        // the override can't leak across threads (env is process-global).
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var(ALLOW_INSECURE_ENV).ok();

        std::env::remove_var(ALLOW_INSECURE_ENV);
        assert!(validate_outbound_url("http://203.0.113.7:9").is_err());
        assert!(validate_outbound_ws_url("ws://203.0.113.7:9").is_err());
        // https/wss always pass; this also kills the ws-wrapper body
        // mutant only-if combined with the remote-reject above.
        assert!(validate_outbound_url("https://ok.example").is_ok());
        assert!(validate_outbound_ws_url("wss://ok.example").is_ok());

        std::env::set_var(ALLOW_INSECURE_ENV, "0"); // not exactly "1"
        assert!(validate_outbound_url("http://203.0.113.7:9").is_err());
        assert!(validate_outbound_ws_url("ws://203.0.113.7:9").is_err());

        std::env::set_var(ALLOW_INSECURE_ENV, "1"); // the override
        assert!(validate_outbound_url("http://203.0.113.7:9").is_ok());
        assert!(validate_outbound_ws_url("ws://203.0.113.7:9").is_ok());

        match prev {
            Some(v) => std::env::set_var(ALLOW_INSECURE_ENV, v),
            None => std::env::remove_var(ALLOW_INSECURE_ENV),
        }
    }
}
