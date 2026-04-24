//! Pool-coordinator wallet (CM-05 slice-2).
//!
//! Loads a secp256k1 private key and provides EIP-1559 transaction
//! signing. Two load paths:
//!
//! 1. **Env var** `CITRATE_POOL_PRIVATE_KEY_HEX` — raw 64-char hex
//!    secp256k1 private key. Suitable for testnet and devnet pilots
//!    where the operator controls the deployment machine. Must be
//!    rotated out before production per the pilot playbook's §5.3
//!    (security posture).
//! 2. **Keystore** (future) — load a Web3 SSv3 keystore (matching
//!    W-01's format) with a passphrase. Reserved but not yet wired;
//!    lands in a follow-up so this file stays focused on signing.
//!
//! Address is derived from the public key via keccak256 of the
//! uncompressed form (Ethereum convention). The derived address
//! MUST match any operator-provided `CITRATE_POOL_WALLET_ADDRESS`
//! config value — mismatch is a load-time error.

use ethereum_types::{H160, H256, U256};
use k256::ecdsa::{signature::hazmat::PrehashSigner, RecoveryId, Signature, SigningKey};
use rlp::RlpStream;
use sha3::{Digest, Keccak256};
use thiserror::Error;

use crate::error::CoordinatorError;

#[derive(Error, Debug)]
pub enum WalletError {
    #[error("private key hex must be 64 characters (got {0})")]
    BadHexLength(usize),
    #[error("invalid hex in private key: {0}")]
    BadHex(String),
    #[error("invalid secp256k1 secret key")]
    BadSecret,
    #[error("operator-declared address does not match derived key ({declared:?} vs {derived:?})")]
    AddressMismatch { declared: H160, derived: H160 },
    #[error("CITRATE_POOL_PRIVATE_KEY_HEX not set")]
    NoKeySource,
}

/// A loaded signing wallet. Cheap to clone (SigningKey is ~32 bytes).
/// Debug prints the address but NEVER the private key material.
#[derive(Clone)]
pub struct Wallet {
    signing_key: SigningKey,
    address: H160,
}

impl std::fmt::Debug for Wallet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wallet")
            .field("address", &self.address)
            .field("signing_key", &"<redacted>")
            .finish()
    }
}

impl Wallet {
    /// Load from the `CITRATE_POOL_PRIVATE_KEY_HEX` env var. Returns
    /// `Err(WalletError::NoKeySource)` if the env var is absent;
    /// other errors describe malformed hex or crypto.
    pub fn from_env() -> Result<Self, WalletError> {
        let hex_key = std::env::var("CITRATE_POOL_PRIVATE_KEY_HEX")
            .map_err(|_| WalletError::NoKeySource)?;
        Self::from_hex(&hex_key)
    }

    /// Build from a hex-encoded private key.
    pub fn from_hex(hex_key: &str) -> Result<Self, WalletError> {
        let trimmed = hex_key.trim().trim_start_matches("0x");
        if trimmed.len() != 64 {
            return Err(WalletError::BadHexLength(trimmed.len()));
        }
        let bytes = hex::decode(trimmed).map_err(|e| WalletError::BadHex(e.to_string()))?;
        let mut sk_bytes = [0u8; 32];
        sk_bytes.copy_from_slice(&bytes);
        let signing_key =
            SigningKey::from_bytes(&sk_bytes.into()).map_err(|_| WalletError::BadSecret)?;
        let address = address_from_signing_key(&signing_key);
        Ok(Self { signing_key, address })
    }

    /// This wallet's EVM address (20 bytes, keccak-derived from
    /// the uncompressed public key).
    pub fn address(&self) -> H160 {
        self.address
    }

    /// Assert the declared address matches the derived one.
    pub fn verify_address(&self, declared: H160) -> Result<(), WalletError> {
        if declared != self.address {
            return Err(WalletError::AddressMismatch {
                declared,
                derived: self.address,
            });
        }
        Ok(())
    }

    /// Sign an EIP-1559 (type-2) transaction. Returns the RLP-encoded
    /// signed tx bytes suitable for `eth_sendRawTransaction`.
    pub fn sign_eip1559(
        &self,
        tx: &Eip1559Tx,
    ) -> Result<Vec<u8>, CoordinatorError> {
        let signing_payload = encode_eip1559_for_signing(tx);
        let digest = keccak256(&signing_payload);
        let (sig, recid) = self
            .signing_key
            .sign_prehash(&digest)
            .map(|s: Signature| {
                let bytes = s.to_bytes();
                let rid = RecoveryId::trial_recovery_from_prehash(
                    &self.signing_key.verifying_key().clone(),
                    &digest,
                    &s,
                )
                .unwrap_or(RecoveryId::from_byte(0).unwrap());
                (bytes, rid.to_byte())
            })
            .map_err(|e| CoordinatorError::Internal(format!("sign: {}", e)))?;

        let r_bytes = &sig[..32];
        let s_bytes = &sig[32..64];
        let r = U256::from_big_endian(r_bytes);
        let s = U256::from_big_endian(s_bytes);
        let y_parity = recid as u64;

        Ok(encode_eip1559_with_signature(tx, y_parity, r, s))
    }
}

/// An EIP-1559 (type-2) transaction we can sign. Minimal surface:
/// no access list, no blob fields — pool-coordinator only needs
/// simple value-carrying + calldata-bearing writes.
#[derive(Debug, Clone)]
pub struct Eip1559Tx {
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee_per_gas: U256,
    pub max_fee_per_gas: U256,
    pub gas_limit: u64,
    pub to: H160,
    pub value: U256,
    pub data: Vec<u8>,
}

fn encode_eip1559_for_signing(tx: &Eip1559Tx) -> Vec<u8> {
    let mut s = RlpStream::new();
    s.begin_list(9);
    s.append(&tx.chain_id);
    s.append(&tx.nonce);
    s.append(&tx.max_priority_fee_per_gas);
    s.append(&tx.max_fee_per_gas);
    s.append(&tx.gas_limit);
    s.append(&tx.to);
    s.append(&tx.value);
    s.append(&tx.data);
    s.begin_list(0); // empty access list
    let mut out = vec![0x02]; // EIP-1559 type prefix
    out.extend_from_slice(&s.out());
    out
}

fn encode_eip1559_with_signature(
    tx: &Eip1559Tx,
    y_parity: u64,
    r: U256,
    s_: U256,
) -> Vec<u8> {
    let mut s = RlpStream::new();
    s.begin_list(12);
    s.append(&tx.chain_id);
    s.append(&tx.nonce);
    s.append(&tx.max_priority_fee_per_gas);
    s.append(&tx.max_fee_per_gas);
    s.append(&tx.gas_limit);
    s.append(&tx.to);
    s.append(&tx.value);
    s.append(&tx.data);
    s.begin_list(0); // empty access list
    s.append(&y_parity);
    s.append(&r);
    s.append(&s_);
    let mut out = vec![0x02];
    out.extend_from_slice(&s.out());
    out
}

fn address_from_signing_key(sk: &SigningKey) -> H160 {
    let vk = sk.verifying_key();
    // Uncompressed SEC1 point = 0x04 || X(32) || Y(32). Drop the
    // leading 0x04 per Ethereum convention, keccak the remaining
    // 64 bytes, take the trailing 20.
    let point = vk.to_encoded_point(false);
    let bytes = &point.as_bytes()[1..]; // skip 0x04
    let h = keccak256(bytes);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&h[12..]);
    H160::from(addr)
}

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let out = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    h
}

/// Derive an EVM tx hash from the signed RLP bytes. Used by the
/// HTTP chain adapter to surface the pre-broadcast hash.
pub fn tx_hash_of_signed(signed_rlp: &[u8]) -> H256 {
    H256::from(keccak256(signed_rlp))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Well-known secp256k1 key for deterministic tests.
    // Address: 0x7e5f4552091a69125d5dfcb7b8c2659029395bdf
    const TEST_HEX: &str =
        "0000000000000000000000000000000000000000000000000000000000000001";

    #[test]
    fn loads_from_hex_without_prefix() {
        let w = Wallet::from_hex(TEST_HEX).expect("load");
        assert_eq!(
            w.address(),
            "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"
                .parse::<H160>()
                .expect("addr")
        );
    }

    #[test]
    fn loads_from_hex_with_0x_prefix() {
        let w = Wallet::from_hex(&format!("0x{}", TEST_HEX)).expect("load");
        assert_eq!(
            w.address(),
            "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"
                .parse::<H160>()
                .expect("addr")
        );
    }

    #[test]
    fn rejects_short_hex() {
        assert!(matches!(
            Wallet::from_hex("abcd").unwrap_err(),
            WalletError::BadHexLength(_)
        ));
    }

    #[test]
    fn rejects_invalid_hex_chars() {
        let bad = "z".repeat(64);
        assert!(matches!(
            Wallet::from_hex(&bad).unwrap_err(),
            WalletError::BadHex(_)
        ));
    }

    #[test]
    fn verify_address_matches_derived() {
        let w = Wallet::from_hex(TEST_HEX).expect("load");
        let declared = "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"
            .parse::<H160>()
            .expect("addr");
        assert!(w.verify_address(declared).is_ok());
    }

    #[test]
    fn verify_address_rejects_mismatch() {
        let w = Wallet::from_hex(TEST_HEX).expect("load");
        let declared = "0x0000000000000000000000000000000000000000"
            .parse::<H160>()
            .expect("addr");
        assert!(matches!(
            w.verify_address(declared).unwrap_err(),
            WalletError::AddressMismatch { .. }
        ));
    }

    #[test]
    fn sign_eip1559_produces_non_empty_rlp_with_type_prefix() {
        let w = Wallet::from_hex(TEST_HEX).expect("load");
        let tx = Eip1559Tx {
            chain_id: 40204,
            nonce: 0,
            max_priority_fee_per_gas: U256::from(1_000_000_000u64),
            max_fee_per_gas: U256::from(2_000_000_000u64),
            gas_limit: 100_000,
            to: H160::repeat_byte(0xAB),
            value: U256::zero(),
            data: vec![0x12, 0x34],
        };
        let signed = w.sign_eip1559(&tx).expect("sign");
        assert_eq!(signed[0], 0x02, "EIP-1559 type prefix");
        assert!(signed.len() > 64);
    }

    #[test]
    fn signed_tx_hash_stable_for_identical_inputs() {
        let w = Wallet::from_hex(TEST_HEX).expect("load");
        let tx = Eip1559Tx {
            chain_id: 40204,
            nonce: 5,
            max_priority_fee_per_gas: U256::from(1u64),
            max_fee_per_gas: U256::from(2u64),
            gas_limit: 21_000,
            to: H160::repeat_byte(0x11),
            value: U256::from(1000u64),
            data: vec![],
        };
        let signed1 = w.sign_eip1559(&tx).expect("sign1");
        let signed2 = w.sign_eip1559(&tx).expect("sign2");
        assert_eq!(tx_hash_of_signed(&signed1), tx_hash_of_signed(&signed2));
    }
}
