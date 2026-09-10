// Semgrep test fixture for fwa-c8-01-outbound-tls-gate.yaml.
// Scan: semgrep scan --config fwa-c8-01-outbound-tls-gate.yaml fwa-c8-01-outbound-tls-gate.rs
// Expect exactly the two `// ruleid:` lines to match; the two `// ok:`
// lines must NOT match.

// ---- ws connect (rule: fwa-c8-01-ws-connect-without-tls-gate) ----

async fn connect_unguarded(self) -> Result<EventStream, CoordinatorError> {
    // ruleid: fwa-c8-01-ws-connect-without-tls-gate
    let (ws_stream, _r) = tokio_tungstenite::connect_async(&self.rpc_ws_url)
        .await
        .map_err(|e| CoordinatorError::Chain(format!("ws connect: {}", e)))?;
    Ok(EventStream::from(ws_stream))
}

async fn connect_guarded(self) -> Result<EventStream, CoordinatorError> {
    crate::outbound::validate_outbound_ws_url(&self.rpc_ws_url)?;
    // ok: fwa-c8-01-ws-connect-without-tls-gate
    let (ws_stream, _r) = tokio_tungstenite::connect_async(&self.rpc_ws_url)
        .await
        .map_err(|e| CoordinatorError::Chain(format!("ws connect: {}", e)))?;
    Ok(EventStream::from(ws_stream))
}

// ---- member endpoint store (rule:
//      fwa-c8-01-member-endpoint-stored-without-tls-gate) ----
// The rule scopes to a fn literally named `parse_endpoints`, mirroring
// the production site in config.rs.

fn parse_endpoints(s: &str) -> Result<HashMap<H160, String>, String> {
    let mut out = HashMap::new();
    let parsed = parse_addr("0x00")?;
    let url = "http://attacker.example/infer";
    // ruleid: fwa-c8-01-member-endpoint-stored-without-tls-gate
    out.insert(parsed, url.to_string());
    Ok(out)
}
