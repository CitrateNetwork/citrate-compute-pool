//! `outbound` — TLS enforcement for the training-worker's outbound
//! HTTP endpoints (CP-B-006, federation graded audit 2026-09-02).
//!
//! The FWA-C8-01 / FWA-BV-CP-01 remediation was applied to
//! `pool-coordinator` but NOT to this crate, even though the worker's
//! outbound clients sign and submit stake and payout transactions. A
//! network position between the worker and its configured RPC
//! (`CITRATE_WORKER_RPC_URL`) or coordinator (`CITRATE_COORDINATOR_URL`)
//! could feed false chain-truth to a daemon that signs money
//! transactions, or (via redirect-follow) cause a signed body to be
//! re-POSTed in cleartext to an off-gate host. This module mirrors
//! `pool-coordinator::outbound` one-for-one (separate workspace crate,
//! so ported rather than shared — the same reason `wallet.rs` is
//! duplicated).
//!
//! The rule:
//! - `https://` — accepted for any host.
//! - `http://`  — accepted **only** for loopback hosts (`localhost`,
//!   `127.0.0.0/8`, `[::1]`). A co-located node stays plain.
//! - anything else — refused, fail closed, at startup so a
//!   misconfigured daemon dies before signing, not mid-job.
//!
//! Dev-only escape hatch: `CITRATE_WORKER_ALLOW_INSECURE_OUTBOUND=1`
//! permits plaintext to non-loopback for LAN test rigs. Logged loudly;
//! never set in production.

/// Env var gating the dev-only plaintext-to-non-loopback escape hatch.
pub const ALLOW_INSECURE_ENV: &str = "CITRATE_WORKER_ALLOW_INSECURE_OUTBOUND";

/// Why an outbound endpoint URL was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutboundUrlError {
    /// Scheme is neither `http` nor `https` (or missing entirely).
    UnsupportedScheme(String),
    /// Plain `http://` to a non-loopback host without the explicit
    /// dev-only override.
    PlaintextNonLoopback(String),
    /// The URL's authority could not be parsed safely (empty host, userinfo).
    Malformed(String),
}

impl core::fmt::Display for OutboundUrlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OutboundUrlError::UnsupportedScheme(u) => write!(
                f,
                "outbound endpoint {u:?} has an unsupported scheme: only https:// (any host) or http:// (loopback) are allowed (CP-B-006)"
            ),
            OutboundUrlError::PlaintextNonLoopback(u) => write!(
                f,
                "outbound endpoint {u:?} is plaintext http:// to a non-loopback host; use https://, or set {ALLOW_INSECURE_ENV}=1 (dev-only) to override (CP-B-006)"
            ),
            OutboundUrlError::Malformed(u) => write!(
                f,
                "outbound endpoint {u:?} has an unparseable or unsafe authority (empty host / userinfo); refusing fail-closed (CP-B-006)"
            ),
        }
    }
}

impl std::error::Error for OutboundUrlError {}

/// Validate `url` as an outbound endpoint, reading the escape hatch
/// from the environment. Called at daemon startup on the RPC + any
/// coordinator/mirror URL.
pub fn validate_outbound_url(url: &str) -> Result<(), OutboundUrlError> {
    let allow_insecure = std::env::var(ALLOW_INSECURE_ENV).is_ok_and(|v| v.trim() == "1");
    validate_outbound_url_with(url, allow_insecure)
}

/// Pure core of [`validate_outbound_url`] (env-free for tests).
pub fn validate_outbound_url_with(url: &str, allow_insecure: bool) -> Result<(), OutboundUrlError> {
    let (scheme, rest) = match url.split_once("://") {
        Some((s, r)) => (s.to_ascii_lowercase(), r),
        None => return Err(OutboundUrlError::UnsupportedScheme(url.to_string())),
    };
    match scheme.as_str() {
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
                    "SECURITY [CP-B-006]: {ALLOW_INSECURE_ENV}=1 — allowing PLAINTEXT outbound to non-loopback {url:?}; dev-only, never use in production"
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
/// fragment, reject userinfo (`@`) and empty hosts (fail closed), strip
/// the port. IPv6 literals keep their brackets stripped.
fn host_of(rest: &str) -> Option<String> {
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    if let Some(v6) = authority.strip_prefix('[') {
        let (inside, after) = v6.split_once(']')?;
        if !(after.is_empty() || after.starts_with(':')) {
            return None;
        }
        return Some(inside.to_string());
    }
    let mut parts = authority.split(':');
    let host = parts.next()?.to_string();
    match (parts.next(), parts.next()) {
        (_, Some(_)) => None,
        (Some(port), None) if port.parse::<u16>().is_err() => None,
        _ if host.is_empty() => None,
        _ => Some(host),
    }
}

/// Is `host` (already stripped of brackets/port) a loopback destination?
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

/// Build a redirect-safe, timeout-bounded reqwest client for the worker's
/// outbound legs (CP-B-006 + CP-B-012).
///
/// - `redirect::Policy::none()` — a 307/308 with `Location: http://<off-gate>`
///   must NOT cause reqwest to re-POST a signed body past the startup gate.
/// - an explicit timeout — a wedged coordinator/RPC cannot hang the poll
///   loop indefinitely.
/// - `.expect()` not `.unwrap_or_default()` — the default client follows
///   redirects, so falling back to it would fail OPEN (CP-B-012).
pub fn redirect_safe_client(timeout: std::time::Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("failed to build redirect-safe outbound HTTP client")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_is_accepted_for_any_host() {
        assert!(validate_outbound_url_with("https://rpc.citrate.ai", false).is_ok());
        assert!(validate_outbound_url_with("https://203.0.113.7:8545/path", false).is_ok());
    }

    #[test]
    fn plaintext_loopback_is_accepted() {
        for url in [
            "http://127.0.0.1:8545",
            "http://127.5.5.5:8080/rpc",
            "http://localhost:8080",
            "http://LOCALHOST:8080",
            "http://[::1]:8545",
        ] {
            assert!(
                validate_outbound_url_with(url, false).is_ok(),
                "{url} rejected"
            );
        }
    }

    // CP-B-006 RED→GREEN: a plaintext remote RPC/coordinator URL — the
    // exact MITM-able money-signing leg — must be refused fail-closed.
    #[test]
    fn plaintext_non_loopback_is_refused() {
        for url in [
            "http://203.0.113.7:8545",
            "http://attacker.example/rpc",
            "http://rpc.citrate.ai",
            "http://localhost.evil.example:8080",
            "http://[2001:db8::1]:8545",
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
            "203.0.113.7:8545",
            "http://",
            "http://user@127.0.0.1:1",
        ] {
            assert!(
                validate_outbound_url_with(url, false).is_err(),
                "{url} accepted"
            );
        }
    }

    #[test]
    fn explicit_insecure_override_allows_plaintext_remote() {
        assert!(validate_outbound_url_with("http://192.168.1.50:8545", true).is_ok());
        assert!(validate_outbound_url_with("ftp://192.168.1.50", true).is_err());
    }
}
