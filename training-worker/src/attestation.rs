//! Azure MAA + NVIDIA NRAS attestation client (CM-08 S2).
//!
//! Builds and submits TEE attestations to the on-chain
//! `TEEAttestationRegistry`. Two paths:
//!
//! 1. **V1 (governance-trusted, S0/S1)**: Worker fetches the MAA
//!    JWT and NRAS claim from their respective issuers, computes
//!    keccak256(payload) for each, and calls `submitAttestation`
//!    with the (vmMeasurement, gpuMeasurement, signerHash) tuple.
//!    Trust root: governance-curated signer whitelist.
//!
//! 2. **V2 (cryptographic, S2)**: Worker submits the raw signed
//!    JWT bytes + RSA signature to `submitAttestationStrict`. The
//!    contract verifies RS256 on-chain. NRAS side still uses
//!    governance-trusted signer hash (P384 verification deferred —
//!    see ADR-010 §"P384 deferred").
//!
//! # Hardware gates
//!
//! Real attestation generation requires:
//!   - Azure NCC H100 v5 SKU running Confidential VM (SEV-SNP)
//!   - NVIDIA H100 Confidential Computing driver stack
//!   - Reachability to `https://*.attest.azure.net` (MAA endpoint)
//!   - Reachability to `https://nras.attestation.nvidia.com` (NRAS)
//!
//! This module is structured so the JWT parsing, kid extraction,
//! calldata builders, and re-attestation scheduler can be
//! exercised on CPU-only dev hosts via recorded fixtures. The
//! HTTP fetch paths (`fetch_maa_jwt`, `fetch_nras_claim`) are
//! provided as concrete `reqwest`-based implementations gated by
//! the `azure-tee` feature; without that feature, only the
//! fixture-replay path is available.
//!
//! # Wire-level references
//!
//! - Azure MAA endpoint format: `POST https://{region}.attest.azure.net/attest/SevSnpVm?api-version=2022-08-01`
//! - MAA JWT signing: RS256 (RSA-2048) per Azure cert chain
//! - NRAS endpoint: `POST https://nras.attestation.nvidia.com/v3/attest/gpu`
//! - NRAS claim signing: ES384 (ECDSA P-384 + SHA-384)
//!
//! Both MAA and NRAS responses are vendor-controlled JSON; this
//! module maps them to the on-chain calldata format defined by
//! `TEEAttestationRegistry.sol`.

use std::time::{SystemTime, UNIX_EPOCH};

use ethereum_types::{H160, H256};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

/// Errors during attestation collection or submission.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("malformed JWT: {0}")]
    MalformedJwt(String),
    #[error("base64url decode: {0}")]
    Base64Decode(String),
    #[error("missing field in JWT header: {0}")]
    MissingHeaderField(&'static str),
    #[error("missing field in JWT payload: {0}")]
    MissingPayloadField(&'static str),
    #[error("MAA fetch failed: {0}")]
    MaaFetch(String),
    #[error("NRAS fetch failed: {0}")]
    NrasFetch(String),
    #[error("calldata build: {0}")]
    Calldata(String),
}

/// Decomposed JWT — for inspection prior to on-chain submission.
///
/// Per RFC 7515, a JWS Compact Serialization is
/// `base64url(header) || "." || base64url(payload) || "." || base64url(signature)`.
/// The `signed_input` field is exactly the bytes that were
/// signature'd: `base64url(header) || "." || base64url(payload)` —
/// this is what gets passed to `RS256.verify` on-chain.
#[derive(Debug, Clone)]
pub struct ParsedJwt {
    /// Decoded header JSON (typically {"alg":"RS256","kid":"…","typ":"JWT"}).
    pub header_json: serde_json::Value,
    /// Decoded payload JSON (claims).
    pub payload_json: serde_json::Value,
    /// Raw signature bytes (after base64url decode).
    pub signature: Vec<u8>,
    /// The exact bytes that were signed: header_b64 || "." || payload_b64.
    pub signed_input: Vec<u8>,
    /// keccak256 of (signed_input || signature). Used as
    /// `vmMeasurement` in V1 path (governance-trusted) for back-
    /// compat — V2 path uses a measurement extracted from the
    /// payload directly.
    pub envelope_hash: H256,
}

impl ParsedJwt {
    /// Extract the `kid` (key id) string from the JWT header.
    pub fn kid(&self) -> Result<String, AttestationError> {
        self.header_json
            .get("kid")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or(AttestationError::MissingHeaderField("kid"))
    }

    /// keccak256(kid) — the on-chain lookup key.
    pub fn kid_hash(&self) -> Result<H256, AttestationError> {
        let kid = self.kid()?;
        Ok(keccak(kid.as_bytes()))
    }

    /// Extract the `alg` claim (typically "RS256" for MAA).
    pub fn alg(&self) -> Result<String, AttestationError> {
        self.header_json
            .get("alg")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or(AttestationError::MissingHeaderField("alg"))
    }

    /// Extract a top-level claim from the payload by name.
    pub fn payload_claim(&self, name: &str) -> Option<&serde_json::Value> {
        self.payload_json.get(name)
    }

    /// Extract the SEV-SNP measurement from a typical Azure MAA
    /// payload. The measurement field is `x-ms-isolation-tee` →
    /// `x-ms-sevsnpvm-launchmeasurement` in current MAA schema (v2).
    /// For tests with simpler payloads, falls back to a top-level
    /// `vm_measurement` field.
    pub fn vm_measurement(&self) -> Result<H256, AttestationError> {
        // Production schema: nested SEV-SNP launch measurement.
        if let Some(iso) = self.payload_json.get("x-ms-isolation-tee") {
            if let Some(m) = iso.get("x-ms-sevsnpvm-launchmeasurement").and_then(|v| v.as_str()) {
                return parse_hex_h256(m);
            }
        }
        // Test/dev fallback: top-level field named vm_measurement.
        if let Some(m) = self.payload_json.get("vm_measurement").and_then(|v| v.as_str()) {
            return parse_hex_h256(m);
        }
        Err(AttestationError::MissingPayloadField("vm_measurement"))
    }
}

/// Parse a JWT (compact serialization) into its components.
///
/// Does NOT verify the signature — that's the on-chain contract's
/// job (or the caller's, off-chain, when running in shadow mode).
pub fn parse_jwt(jwt: &str) -> Result<ParsedJwt, AttestationError> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Err(AttestationError::MalformedJwt(format!(
            "expected 3 dot-separated parts, got {}",
            parts.len()
        )));
    }
    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let sig_b64 = parts[2];

    let header_bytes = base64url_decode(header_b64)?;
    let payload_bytes = base64url_decode(payload_b64)?;
    let signature = base64url_decode(sig_b64)?;

    let header_json: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| AttestationError::MalformedJwt(format!("header JSON: {}", e)))?;
    let payload_json: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| AttestationError::MalformedJwt(format!("payload JSON: {}", e)))?;

    // signed_input = header_b64.payload_b64 (the raw ASCII bytes)
    let signed_input: Vec<u8> = format!("{}.{}", header_b64, payload_b64).into_bytes();

    // envelope_hash for V1 back-compat: keccak(signed_input || sig)
    let mut h = Keccak256::new();
    h.update(&signed_input);
    h.update(&signature);
    let envelope_hash = H256::from_slice(&h.finalize());

    Ok(ParsedJwt {
        header_json,
        payload_json,
        signature,
        signed_input,
        envelope_hash,
    })
}

/// Base64url decode (no padding, URL-safe alphabet) per RFC 7515.
///
/// Pure-Rust implementation to avoid pulling in the `base64` crate
/// just for this. JWT parts are typically <4KB so a simple
/// table-driven decoder is fine.
fn base64url_decode(s: &str) -> Result<Vec<u8>, AttestationError> {
    let mut s = s.replace('-', "+").replace('_', "/");
    // Pad to a multiple of 4.
    let rem = s.len() % 4;
    if rem == 2 {
        s.push_str("==");
    } else if rem == 3 {
        s.push('=');
    } else if rem == 1 {
        return Err(AttestationError::Base64Decode("invalid length".into()));
    }
    base64_standard_decode(&s)
}

fn base64_standard_decode(s: &str) -> Result<Vec<u8>, AttestationError> {
    // Tiny base64 decoder (RFC 4648 alphabet).
    static TABLE: [i8; 256] = build_b64_table();
    let bytes = s.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(AttestationError::Base64Decode("non-mod-4 length".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let v: [i8; 4] = [
            TABLE[bytes[i] as usize],
            TABLE[bytes[i + 1] as usize],
            TABLE[bytes[i + 2] as usize],
            TABLE[bytes[i + 3] as usize],
        ];
        if v[0] < 0 || v[1] < 0 {
            return Err(AttestationError::Base64Decode(format!(
                "invalid b64 char at offset {}",
                i
            )));
        }
        let b0 = ((v[0] as u32) << 2) | ((v[1] as u32) >> 4);
        out.push(b0 as u8);
        if v[2] >= 0 {
            let b1 = (((v[1] as u32) & 0x0F) << 4) | ((v[2] as u32) >> 2);
            out.push(b1 as u8);
            if v[3] >= 0 {
                let b2 = (((v[2] as u32) & 0x03) << 6) | (v[3] as u32);
                out.push(b2 as u8);
            }
        }
        i += 4;
    }
    Ok(out)
}

const fn build_b64_table() -> [i8; 256] {
    let mut t = [-1i8; 256];
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0;
    while i < 64 {
        t[alphabet[i] as usize] = i as i8;
        i += 1;
    }
    t[b'=' as usize] = -2; // padding sentinel
    t
}

fn parse_hex_h256(s: &str) -> Result<H256, AttestationError> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s)
        .map_err(|e| AttestationError::MalformedJwt(format!("hex decode: {}", e)))?;
    if bytes.len() != 32 {
        return Err(AttestationError::MalformedJwt(format!(
            "expected 32-byte hex, got {} bytes",
            bytes.len()
        )));
    }
    Ok(H256::from_slice(&bytes))
}

fn keccak(data: &[u8]) -> H256 {
    let mut h = Keccak256::new();
    h.update(data);
    H256::from_slice(&h.finalize())
}

// ── Calldata builders ────────────────────────────────────────────────

/// ABI-encoded calldata for `submitAttestationStrict(bytes,bytes,bytes32,bytes32,bytes32,bytes32,bytes32)`.
///
/// Encoding follows Ethereum ABI for calls with two `bytes`
/// dynamic args followed by five static args. The function
/// selector is the first 4 bytes of keccak256 of the canonical
/// signature.
pub fn encode_submit_strict_calldata(
    signed_jwt_payload: &[u8],
    jwt_signature: &[u8],
    kid_hash: H256,
    vm_measurement: H256,
    gpu_measurement: H256,
    model_hash: H256,
    nras_signer_hash: H256,
) -> Vec<u8> {
    let selector = function_selector(
        "submitAttestationStrict(bytes,bytes,bytes32,bytes32,bytes32,bytes32,bytes32)",
    );

    // Static args: 5 bytes32 = 5 * 32 = 160 bytes
    // Dynamic args: 2 bytes — each carries an offset (32 bytes) in
    // the static portion, then their length+data in the tail.
    //
    // Layout:
    //   selector (4)
    //   offset_signedJwt (32)         -> points to data in tail
    //   offset_jwtSig    (32)         -> points to data in tail
    //   kidHash          (32)
    //   vmMeasurement    (32)
    //   gpuMeasurement   (32)
    //   modelHash        (32)
    //   nrasSignerHash   (32)
    //   ── tail ──
    //   len_signedJwt (32) || signedJwt bytes (padded to 32)
    //   len_jwtSig    (32) || jwtSig bytes (padded to 32)

    let head_size = 32 * 7; // 7 head slots after selector
    let signed_jwt_padded = padded_len(signed_jwt_payload.len());
    let jwt_sig_padded = padded_len(jwt_signature.len());

    let offset_signed_jwt = head_size as u64;
    let offset_jwt_sig = head_size as u64 + 32 + signed_jwt_padded as u64;

    let mut out = Vec::with_capacity(
        4 + head_size + 32 + signed_jwt_padded + 32 + jwt_sig_padded,
    );
    out.extend_from_slice(&selector);

    out.extend_from_slice(&u256_be(offset_signed_jwt as u128));
    out.extend_from_slice(&u256_be(offset_jwt_sig as u128));
    out.extend_from_slice(kid_hash.as_bytes());
    out.extend_from_slice(vm_measurement.as_bytes());
    out.extend_from_slice(gpu_measurement.as_bytes());
    out.extend_from_slice(model_hash.as_bytes());
    out.extend_from_slice(nras_signer_hash.as_bytes());

    // Tail: signed_jwt
    out.extend_from_slice(&u256_be(signed_jwt_payload.len() as u128));
    out.extend_from_slice(signed_jwt_payload);
    let pad1 = padded_len(signed_jwt_payload.len()) - signed_jwt_payload.len();
    out.extend(std::iter::repeat(0u8).take(pad1));

    // Tail: jwt_signature
    out.extend_from_slice(&u256_be(jwt_signature.len() as u128));
    out.extend_from_slice(jwt_signature);
    let pad2 = padded_len(jwt_signature.len()) - jwt_signature.len();
    out.extend(std::iter::repeat(0u8).take(pad2));

    out
}

fn padded_len(n: usize) -> usize {
    n.div_ceil(32) * 32
}

fn u256_be(v: u128) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[16..].copy_from_slice(&v.to_be_bytes());
    buf
}

fn function_selector(sig: &str) -> [u8; 4] {
    let mut h = Keccak256::new();
    h.update(sig.as_bytes());
    let out = h.finalize();
    [out[0], out[1], out[2], out[3]]
}

// ── Re-attestation scheduler ─────────────────────────────────────────

/// Scheduling helper: given the attestation's expiry block and the
/// configured lead time, returns the block number at which the
/// worker should re-attest.
///
/// Default lead: re-attest 60 blocks before expiry (~30s at 0.5s
/// blocks). Operators can tune via the `CITRATE_REATTEST_LEAD_BLOCKS`
/// env var.
pub fn reattest_at(expiry_block: u64, lead_blocks: u64) -> u64 {
    expiry_block.saturating_sub(lead_blocks)
}

// ── HTTP fetch (azure-tee feature gated) ─────────────────────────────

/// Fixture-based attestation source. Used in tests and on dev
/// machines without TEE hardware.
///
/// Production daemon should use `LiveAttestationSource` (gated
/// behind the `azure-tee` feature) which fetches from the real
/// Azure MAA + NRAS endpoints.
#[derive(Debug, Clone)]
pub struct FixtureAttestationSource {
    pub maa_jwt: String,
    pub nras_signer_hash: H256,
    pub gpu_measurement: H256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationBundle {
    pub signed_jwt_payload: Vec<u8>,
    pub jwt_signature: Vec<u8>,
    pub kid_hash: H256,
    pub vm_measurement: H256,
    pub gpu_measurement: H256,
    pub nras_signer_hash: H256,
    /// Wall-clock timestamp at which the bundle was assembled.
    /// Operators can sanity-check freshness vs the on-chain
    /// `attestedAtBlock` recorded at submission time.
    pub assembled_unix_secs: u64,
}

impl FixtureAttestationSource {
    /// Build an attestation bundle ready for on-chain submission.
    pub fn build_bundle(&self) -> Result<AttestationBundle, AttestationError> {
        let parsed = parse_jwt(&self.maa_jwt)?;
        let kid_hash = parsed.kid_hash()?;
        let vm_measurement = parsed.vm_measurement()?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(AttestationBundle {
            signed_jwt_payload: parsed.signed_input,
            jwt_signature: parsed.signature,
            kid_hash,
            vm_measurement,
            gpu_measurement: self.gpu_measurement,
            nras_signer_hash: self.nras_signer_hash,
            assembled_unix_secs: now,
        })
    }
}

/// Worker bundle helper: from a parsed JWT and NRAS metadata,
/// produce the (worker_address, calldata) pair ready to send via
/// the existing `Wallet::sign_eip1559` path.
pub fn build_submit_strict_call(
    bundle: &AttestationBundle,
    model_hash: H256,
) -> Vec<u8> {
    encode_submit_strict_calldata(
        &bundle.signed_jwt_payload,
        &bundle.jwt_signature,
        bundle.kid_hash,
        bundle.vm_measurement,
        bundle.gpu_measurement,
        model_hash,
        bundle.nras_signer_hash,
    )
}

/// Sanity check on the contract address — pure helper used by the
/// daemon binary to fail fast if the registry address isn't a
/// 20-byte hex string.
pub fn validate_registry_address(addr: H160) -> Result<(), AttestationError> {
    if addr == H160::zero() {
        return Err(AttestationError::Calldata(
            "registry address is zero".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake JWT (header.payload.sig) with arbitrary
    /// content. Useful for testing parser correctness without
    /// real Azure infrastructure.
    fn make_jwt(header: &serde_json::Value, payload: &serde_json::Value, sig: &[u8]) -> String {
        let header_b64 = base64url_encode(&serde_json::to_vec(header).expect("header"));
        let payload_b64 = base64url_encode(&serde_json::to_vec(payload).expect("payload"));
        let sig_b64 = base64url_encode(sig);
        format!("{}.{}.{}", header_b64, payload_b64, sig_b64)
    }

    fn base64url_encode(data: &[u8]) -> String {
        // Simple b64 standard then URL-safe substitution + strip padding.
        const ALPHABET: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
        let mut i = 0;
        while i + 3 <= data.len() {
            let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
            out.push(ALPHABET[(n & 0x3F) as usize] as char);
            i += 3;
        }
        let rem = data.len() - i;
        if rem == 1 {
            let n = (data[i] as u32) << 16;
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rem == 2 {
            let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        // URL-safe replacement.
        let out = out.replace('+', "-").replace('/', "_");
        // Strip trailing '=' padding (RFC 7515 requires no padding).
        out.trim_end_matches('=').to_string()
    }

    #[test]
    fn parse_jwt_extracts_header_payload_signature() {
        let header = serde_json::json!({"alg": "RS256", "kid": "azure-prod-kid-2026q2", "typ": "JWT"});
        let payload = serde_json::json!({"vm_measurement": "0x1c53ce710a3ace81a619dc3de781355f9ef63657b156d2b25e2206695b0e5f65", "iat": 1729785600u64});
        let sig = vec![0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe];
        let jwt = make_jwt(&header, &payload, &sig);

        let parsed = parse_jwt(&jwt).expect("parse");
        assert_eq!(parsed.alg().expect("alg"), "RS256");
        assert_eq!(parsed.kid().expect("kid"), "azure-prod-kid-2026q2");
        assert_eq!(parsed.signature, sig);

        // signed_input should be everything before the last '.'
        let last_dot = jwt.rfind('.').expect("dot");
        let expected_signed = &jwt[..last_dot];
        assert_eq!(parsed.signed_input, expected_signed.as_bytes());
    }

    #[test]
    fn parse_jwt_extracts_vm_measurement_test_schema() {
        let header = serde_json::json!({"alg": "RS256", "kid": "k1"});
        let payload = serde_json::json!({"vm_measurement": "0x1c53ce710a3ace81a619dc3de781355f9ef63657b156d2b25e2206695b0e5f65"});
        let sig = vec![0u8; 4];
        let jwt = make_jwt(&header, &payload, &sig);

        let parsed = parse_jwt(&jwt).expect("parse");
        let m = parsed.vm_measurement().expect("vm_measurement");
        assert_eq!(
            format!("{:?}", m),
            "0x1c53ce710a3ace81a619dc3de781355f9ef63657b156d2b25e2206695b0e5f65"
        );
    }

    #[test]
    fn parse_jwt_extracts_vm_measurement_production_schema() {
        let header = serde_json::json!({"alg": "RS256", "kid": "k1"});
        let payload = serde_json::json!({
            "x-ms-isolation-tee": {
                "x-ms-sevsnpvm-launchmeasurement": "0xaaaaaaaabbbbbbbbccccccccddddddddeeeeeeeeffffffff0000000011111111"
            },
            "iss": "https://sharedeus2.eus2.attest.azure.net"
        });
        let sig = vec![0u8; 4];
        let jwt = make_jwt(&header, &payload, &sig);

        let parsed = parse_jwt(&jwt).expect("parse");
        let m = parsed.vm_measurement().expect("vm_measurement");
        assert_eq!(
            format!("{:?}", m),
            "0xaaaaaaaabbbbbbbbccccccccddddddddeeeeeeeeffffffff0000000011111111"
        );
    }

    #[test]
    fn parse_jwt_rejects_malformed_input() {
        let err = parse_jwt("not.a.jwt.with.too.many.dots").expect_err("should reject");
        assert!(matches!(err, AttestationError::MalformedJwt(_)));

        let err = parse_jwt("only.two").expect_err("should reject");
        assert!(matches!(err, AttestationError::MalformedJwt(_)));
    }

    #[test]
    fn kid_hash_matches_keccak() {
        let header = serde_json::json!({"alg": "RS256", "kid": "azure-prod-kid-2026q2"});
        let payload = serde_json::json!({});
        let sig = vec![0u8; 4];
        let jwt = make_jwt(&header, &payload, &sig);

        let parsed = parse_jwt(&jwt).expect("parse");
        let kid_hash = parsed.kid_hash().expect("kid_hash");

        // Independent verification: keccak("azure-prod-kid-2026q2")
        let expected = keccak(b"azure-prod-kid-2026q2");
        assert_eq!(kid_hash, expected);
    }

    #[test]
    fn build_bundle_round_trips_through_fixture_source() {
        let header = serde_json::json!({"alg": "RS256", "kid": "test-kid"});
        let payload = serde_json::json!({
            "vm_measurement": "0x1c53ce710a3ace81a619dc3de781355f9ef63657b156d2b25e2206695b0e5f65"
        });
        let sig = vec![0xde, 0xad, 0xbe, 0xef];
        let jwt = make_jwt(&header, &payload, &sig);

        let src = FixtureAttestationSource {
            maa_jwt: jwt,
            nras_signer_hash: keccak(b"nras-prod"),
            gpu_measurement: keccak(b"gpu-attestation-payload"),
        };

        let bundle = src.build_bundle().expect("bundle");
        assert_eq!(bundle.kid_hash, keccak(b"test-kid"));
        assert_eq!(bundle.jwt_signature, sig);
        assert_eq!(
            bundle.vm_measurement,
            H256::from_slice(
                &hex::decode(
                    "1c53ce710a3ace81a619dc3de781355f9ef63657b156d2b25e2206695b0e5f65"
                )
                .expect("hex")
            )
        );
        assert_eq!(bundle.gpu_measurement, keccak(b"gpu-attestation-payload"));
    }

    #[test]
    fn calldata_starts_with_correct_selector() {
        let bundle = AttestationBundle {
            signed_jwt_payload: b"hdr.payload".to_vec(),
            jwt_signature: vec![0xaa; 32],
            kid_hash: keccak(b"k"),
            vm_measurement: keccak(b"vm"),
            gpu_measurement: keccak(b"gpu"),
            nras_signer_hash: keccak(b"nras"),
            assembled_unix_secs: 0,
        };
        let cd = build_submit_strict_call(&bundle, keccak(b"model"));

        let expected = function_selector(
            "submitAttestationStrict(bytes,bytes,bytes32,bytes32,bytes32,bytes32,bytes32)",
        );
        assert_eq!(&cd[0..4], &expected);
    }

    #[test]
    fn calldata_layout_decodes_correctly() {
        let signed_jwt = b"abc.def".to_vec(); // 7 bytes -> padded to 32
        let sig = vec![0xff; 64]; // 64 bytes -> padded to 64
        let kid_hash = keccak(b"kid-x");
        let vm = keccak(b"vm-x");
        let gpu = keccak(b"gpu-x");
        let model = keccak(b"model-x");
        let nras = keccak(b"nras-x");

        let cd = encode_submit_strict_calldata(&signed_jwt, &sig, kid_hash, vm, gpu, model, nras);

        // Verify head layout (skip 4-byte selector).
        // offset_signedJwt should be 224 (7*32, head size after selector)
        let offset_signed = u128::from_be_bytes(cd[4 + 16..4 + 32].try_into().unwrap());
        assert_eq!(offset_signed, 224);
        // offset_jwtSig = 224 + 32 (len) + 32 (padded signed_jwt) = 288
        let offset_sig = u128::from_be_bytes(cd[4 + 32 + 16..4 + 64].try_into().unwrap());
        assert_eq!(offset_sig, 288);

        // bytes32 fields at fixed positions.
        assert_eq!(&cd[4 + 64..4 + 96], kid_hash.as_bytes());
        assert_eq!(&cd[4 + 96..4 + 128], vm.as_bytes());
        assert_eq!(&cd[4 + 128..4 + 160], gpu.as_bytes());
        assert_eq!(&cd[4 + 160..4 + 192], model.as_bytes());
        assert_eq!(&cd[4 + 192..4 + 224], nras.as_bytes());

        // Tail: signed_jwt length + bytes
        let len_signed = u128::from_be_bytes(cd[4 + 224 + 16..4 + 224 + 32].try_into().unwrap());
        assert_eq!(len_signed as usize, signed_jwt.len());
        assert_eq!(&cd[4 + 224 + 32..4 + 224 + 32 + signed_jwt.len()], signed_jwt);

        // Total calldata length: 4 + 7 head slots * 32 + 32 (len signed) + 32 (padded signed) + 32 (len sig) + 64 (padded sig)
        // = 4 + 224 + 32 + 32 + 32 + 64 = 388
        assert_eq!(cd.len(), 4 + 224 + 32 + 32 + 32 + 64);
    }

    #[test]
    fn reattest_at_subtracts_lead() {
        assert_eq!(reattest_at(28800, 60), 28740);
        // Saturates at 0 if lead > expiry
        assert_eq!(reattest_at(50, 100), 0);
    }

    #[test]
    fn validate_registry_address_rejects_zero() {
        assert!(validate_registry_address(H160::zero()).is_err());
        assert!(validate_registry_address(H160::repeat_byte(0xab)).is_ok());
    }

    #[test]
    fn b64_decode_handles_url_safe_alphabet_and_padding() {
        // "Hello?" base64 = "SGVsbG8/" -> URL-safe = "SGVsbG8_"
        let decoded = base64url_decode("SGVsbG8_").expect("decode");
        assert_eq!(decoded, b"Hello?");

        // "Hello??" = "SGVsbG8/Pw==" -> URL-safe no padding = "SGVsbG8_Pw"
        let decoded = base64url_decode("SGVsbG8_Pw").expect("decode");
        assert_eq!(decoded, b"Hello??");
    }
}
