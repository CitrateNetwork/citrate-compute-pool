// Negative fixture: the GUARDED parse_endpoints must NOT match
// fwa-c8-01-member-endpoint-stored-without-tls-gate.
// Scan and assert zero findings on this file.

fn parse_endpoints(s: &str) -> Result<HashMap<H160, String>, String> {
    let mut out = HashMap::new();
    let parsed = parse_addr("0x00")?;
    let url = "https://m1.pool.example/infer";
    validate_outbound_url(url)?;
    // ok: fwa-c8-01-member-endpoint-stored-without-tls-gate
    out.insert(parsed, url.to_string());
    Ok(out)
}
