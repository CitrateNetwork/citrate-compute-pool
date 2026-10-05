//! Federated LoRA rounds (HUP-S9.2): the worker and coordinator halves of one
//! cluster-scoped paraconsensus round.
//!
//! A round, end to end:
//!
//! 1. The cluster coordinator publishes one `lora_delta` job per roster device,
//!    each carrying the same [`round::RoundConfig`] and round ordinal.
//! 2. A worker trains the round's start adapter on its own verified, redacted
//!    trajectories (the S9.3 export) through an operator-configured trainer
//!    ([`trainer`]), then differences the trained adapter against the start
//!    adapter tensor by tensor ([`delta`]), encodes the difference on the fixed
//!    Q16 grid and commits to it chunk by chunk ([`tree`]).
//! 3. The worker uploads the delta artifact to the coordinator and submits a
//!    result whose inner signature binds the round, the worker and the delta
//!    root, so the round bundle is verifiable without trusting the coordinator.
//! 4. The coordinator builds one `0x0110` Belnap input per chunk ([`belnap`]),
//!    runs it through the precompile on a Citrate node, and commits the input,
//!    output and participant roots to `FederatedRoundLedger` on chain.
//! 5. Anyone can replay the round from the bundle and the delta artifacts, and
//!    within the challenge window prove a wrong chunk on chain, where the ledger
//!    recomputes it through the same precompile.
//!
//! Wire formats are specified in citrate-chain `docs/fl/FL_ROUND_V1.md`; the
//! chain's replay tool is a second, independent implementation of them.

pub mod belnap;
pub mod dataset;
pub mod delta;
pub mod gguf;
pub mod round;
pub mod runner;
pub mod trainer;
pub mod tree;

/// A 32-byte word: round ids, roots, digests.
pub type B32 = [u8; 32];
/// A 20-byte EVM address.
pub type Addr = [u8; 20];

/// `keccak256` over the concatenation of `parts`.
pub fn keccak(parts: &[&[u8]]) -> B32 {
    use sha3::{Digest, Keccak256};
    let mut h = Keccak256::new();
    for p in parts {
        h.update(p);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

/// `sha256` of a byte slice.
pub fn sha256(bytes: &[u8]) -> B32 {
    use sha2::{Digest, Sha256};
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(bytes));
    out
}

/// Stream a file through `sha256` without holding it in memory.
pub fn sha256_file(path: &std::path::Path) -> std::io::Result<B32> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    Ok(out)
}

/// `0x`-prefixed lowercase hex.
pub fn hex0x(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}

/// Parse a fixed-size hex value with or without `0x`.
pub fn parse_hex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    let h = s.trim();
    let h = h.strip_prefix("0x").unwrap_or(h);
    let v = hex::decode(h).map_err(|e| format!("{s:?} is not hex: {e}"))?;
    v.try_into()
        .map_err(|v: Vec<u8>| format!("{s:?} is {} bytes, expected {N}", v.len()))
}

/// Serde helpers for fixed-size byte arrays as `0x` hex strings.
pub mod hexser {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer, const N: usize>(v: &[u8; N], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&super::hex0x(v))
    }

    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(
        d: D,
    ) -> Result<[u8; N], D::Error> {
        let s = String::deserialize(d)?;
        super::parse_hex::<N>(&s).map_err(serde::de::Error::custom)
    }

    /// The same, for a list of addresses.
    pub mod vec20 {
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S: Serializer>(v: &[[u8; 20]], s: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut seq = s.serialize_seq(Some(v.len()))?;
            for a in v {
                seq.serialize_element(&super::super::hex0x(a))?;
            }
            seq.end()
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[u8; 20]>, D::Error> {
            let v = Vec::<String>::deserialize(d)?;
            v.iter()
                .map(|s| super::super::parse_hex::<20>(s).map_err(serde::de::Error::custom))
                .collect()
        }
    }
}
