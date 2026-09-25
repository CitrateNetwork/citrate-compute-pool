//! Training-worker wallet (S1 HTTP-chain slice).
//!
//! FIXME(post-pilot): unify with pool-coordinator/src/wallet.rs by
//! moving both to citrate-wallet-core or a new citrate-evm-wallet
//! crate. Duplicated today to keep the S1 HTTP-chain-client work
//! unblocked. The two copies MUST stay byte-for-byte compatible on
//! the Web3 SSv3 keystore format — a keystore created by the pool-
//! coordinator daemon must unlock under the training-worker daemon
//! and vice versa. If you edit signing or keystore handling in one
//! copy, edit it in the other until the unification lands. The
//! `SHARED-KEYSTORE` regions (keystore decryption, KDF floor, env-secret
//! handling) are enforced byte-identical by
//! `the_two_wallet_copies_share_one_keystore_implementation` (PBA-L4-010).
//!
//! Loads a secp256k1 private key and provides EIP-1559 transaction
//! signing. Two load paths:
//!
//! 1. **Env var** `CITRATE_TRAINING_PRIVATE_KEY_HEX` — raw 64-char hex
//!    secp256k1 private key. Suitable for testnet and devnet pilots
//!    where the operator controls the deployment machine. Must be
//!    rotated out before production.
//! 2. **Keystore** — load a Web3 SSv3 keystore (matching the JS SDK's
//!    W-01 format) with a passphrase.
//!
//! Address is derived from the public key via keccak256 of the
//! uncompressed form (Ethereum convention).

use ethereum_types::{H160, H256, U256};
use k256::ecdsa::{signature::hazmat::PrehashSigner, RecoveryId, Signature, SigningKey};
use rlp::RlpStream;
use serde::Deserialize;
use sha3::{Digest, Keccak256};
use thiserror::Error;
use zeroize::Zeroizing;

// ── Web3 SSv3 keystore JSON shape ───────────────────────────────

#[derive(Debug, Deserialize)]
struct KeystoreFile {
    version: u32,
    address: Option<String>,
    crypto: KeystoreCrypto,
}

#[derive(Debug, Deserialize)]
struct KeystoreCrypto {
    cipher: String,
    cipherparams: KeystoreCipherParams,
    ciphertext: String,
    kdf: String,
    kdfparams: KeystoreKdfParams,
    mac: String,
}

#[derive(Debug, Deserialize)]
struct KeystoreCipherParams {
    iv: String,
}

#[derive(Debug, Deserialize)]
struct KeystoreKdfParams {
    prf: String,
    c: u32,
    salt: String,
    dklen: u32,
}

const ENV_KEYSTORE_PATH: &str = "CITRATE_TRAINING_KEYSTORE_PATH";
const ENV_KEYSTORE_PASSPHRASE: &str = "CITRATE_TRAINING_KEYSTORE_PASSPHRASE";
const ENV_PRIVATE_KEY_HEX: &str = "CITRATE_TRAINING_PRIVATE_KEY_HEX";

// BEGIN SHARED-KEYSTORE (PBA-L4-010) -- keep byte-identical with the other
// copy (pool-coordinator/src/wallet.rs <-> training-worker/src/wallet.rs); the
// `the_two_wallet_copies_share_one_keystore_implementation` test enforces it.

/// Hard floor for the keystore's PBKDF2-HMAC-SHA256 iteration count. Below it
/// the passphrase is close to unprotected against an offline guess (at `c=1`
/// one GPU tries billions of passphrases a second), so such a keystore is
/// refused rather than loaded with a warning. 10,000 is the oldest count any
/// Citrate SDK has minted; the modern floor below only warns.
pub const MIN_PBKDF2_ITERS: u32 = 10_000;

/// Modern floor for PBKDF2-HMAC-SHA256 iteration counts (OWASP 2023
/// guidance is 600k for HMAC-SHA256). Loading a legacy keystore below
/// this floor is still permitted (backward compat, ENCRYPT-S1 WP-10)
/// but emits a soft warning so operators know to re-mint at rotation.
const MIN_MODERN_PBKDF2_ITERS: u32 = 600_000;

/// Read a secret from the environment and remove it (PBA-L4-010).
///
/// The value is held in `Zeroizing` so this copy is wiped on drop, and the
/// variable is removed so it is not inherited by child processes or read again
/// later by anything else in the process. This does not scrub the kernel's
/// copy of the initial environment (`/proc/<pid>/environ` on Linux); prefer the
/// keystore path, and the raw-hex path only on testnets. Call it at startup,
/// before other threads read the environment.
fn take_env_secret(name: &str) -> Option<Zeroizing<String>> {
    let value = std::env::var(name).ok().map(Zeroizing::new);
    if value.is_some() {
        std::env::remove_var(name);
    }
    value
}

/// Refuse a keystore whose KDF cost is below [`MIN_PBKDF2_ITERS`]. Below the
/// modern floor it loads, and the count is returned so the caller can warn once
/// logging is up (a key is loaded before the tracing subscriber exists, so a
/// warning emitted here would be lost).
fn check_kdf_cost(iterations: u32) -> Result<Option<u32>, WalletError> {
    if iterations < MIN_PBKDF2_ITERS {
        return Err(WalletError::WeakKdf {
            iterations,
            floor: MIN_PBKDF2_ITERS,
        });
    }
    // ENCRYPT-S1 WP-10: non-breaking hardening signal. Legacy
    // keystores below the modern iteration floor still load (we do
    // NOT reject — that would break backward compat); the caller warns
    // (Wallet::log_load_warnings) so the key gets re-minted at rotation.
    Ok((iterations < MIN_MODERN_PBKDF2_ITERS).then_some(iterations))
}
// END SHARED-KEYSTORE

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
    #[error("no key source configured (set CITRATE_TRAINING_PRIVATE_KEY_HEX or CITRATE_TRAINING_KEYSTORE_PATH+CITRATE_TRAINING_KEYSTORE_PASSPHRASE)")]
    NoKeySource,
    #[error(
        "CITRATE_TRAINING_KEYSTORE_PASSPHRASE required when CITRATE_TRAINING_KEYSTORE_PATH is set"
    )]
    MissingPassphrase,
    #[error("keystore I/O: {0}")]
    KeystoreIo(String),
    #[error("keystore parse: {0}")]
    KeystoreParse(String),
    #[error("unsupported keystore version: {0} (only v3 is supported)")]
    UnsupportedKeystoreVersion(u32),
    #[error("unsupported cipher: {0} (only aes-128-ctr is supported)")]
    UnsupportedCipher(String),
    #[error("unsupported kdf: {0} (only pbkdf2 is supported)")]
    UnsupportedKdf(String),
    #[error("unsupported kdf prf: {0} (only hmac-sha256 is supported)")]
    UnsupportedKdfPrf(String),
    #[error("unsupported kdf dklen: {0} (only 32 is supported)")]
    UnsupportedKdfDklen(u32),
    #[error("invalid passphrase")]
    InvalidPassphrase,
    #[error("keystore PBKDF2 iteration count {iterations} is below the floor of {floor}; re-mint the keystore")]
    WeakKdf { iterations: u32, floor: u32 },
    #[error("sign: {0}")]
    Sign(String),
}

/// A loaded signing wallet. Cheap to clone (SigningKey is ~32 bytes).
/// Debug prints the address but NEVER the private key material.
#[derive(Clone)]
pub struct Wallet {
    signing_key: SigningKey,
    address: H160,
    /// PBKDF2 iteration count of the keystore this was loaded from, when it
    /// is below the modern floor. Reported by [`Wallet::log_load_warnings`].
    below_modern_kdf: Option<u32>,
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
    /// Load from env. Prefers the keystore path
    /// (`CITRATE_TRAINING_KEYSTORE_PATH` +
    /// `CITRATE_TRAINING_KEYSTORE_PASSPHRASE`); falls back to raw hex
    /// (`CITRATE_TRAINING_PRIVATE_KEY_HEX`). The raw-hex path is
    /// testnet-only.
    ///
    /// Returns `Err(WalletError::NoKeySource)` if neither variable
    /// set is present.
    ///
    /// Removes the secret variables from the environment, which is only sound
    /// while the process is single-threaded: call it before starting any
    /// thread or async runtime.
    pub fn from_env() -> Result<Self, WalletError> {
        // BEGIN SHARED-KEYSTORE (PBA-L4-010)
        if let Ok(path) = std::env::var(ENV_KEYSTORE_PATH) {
            let passphrase = take_env_secret(ENV_KEYSTORE_PASSPHRASE);
            // Never leave a raw key behind next to a keystore, even when the
            // passphrase is missing and this returns an error.
            drop(take_env_secret(ENV_PRIVATE_KEY_HEX));
            let passphrase = passphrase.ok_or(WalletError::MissingPassphrase)?;
            return Self::from_keystore(&path, &passphrase);
        }
        if let Some(hex_key) = take_env_secret(ENV_PRIVATE_KEY_HEX) {
            return Self::from_hex(&hex_key);
        }
        Err(WalletError::NoKeySource)
        // END SHARED-KEYSTORE
    }

    /// Load from a Web3 Secret Storage v3 keystore file. Matches the
    /// JS SDK's W-01 keystore format byte-for-byte:
    ///   - PBKDF2-HMAC-SHA256 key derivation (32 bytes output)
    ///   - AES-128-CTR decryption (key = dkey[0..16], iv per keystore)
    ///   - Keccak-256 MAC over (dkey[16..32] || ciphertext)
    ///
    /// Keystores are portable: a key created in the JS CitrateWallet
    /// can be unlocked here and vice versa.
    pub fn from_keystore(path: &str, passphrase: &str) -> Result<Self, WalletError> {
        // BEGIN SHARED-KEYSTORE (PBA-L4-010)
        let raw = std::fs::read_to_string(path)
            .map_err(|e| WalletError::KeystoreIo(format!("read {}: {}", path, e)))?;
        let file: KeystoreFile =
            serde_json::from_str(&raw).map_err(|e| WalletError::KeystoreParse(e.to_string()))?;
        if file.version != 3 {
            return Err(WalletError::UnsupportedKeystoreVersion(file.version));
        }

        let crypto = &file.crypto;
        if crypto.cipher != "aes-128-ctr" {
            return Err(WalletError::UnsupportedCipher(crypto.cipher.clone()));
        }
        if crypto.kdf != "pbkdf2" {
            return Err(WalletError::UnsupportedKdf(crypto.kdf.clone()));
        }
        if crypto.kdfparams.prf != "hmac-sha256" {
            return Err(WalletError::UnsupportedKdfPrf(crypto.kdfparams.prf.clone()));
        }
        if crypto.kdfparams.dklen != 32 {
            return Err(WalletError::UnsupportedKdfDklen(crypto.kdfparams.dklen));
        }

        let salt = hex::decode(&crypto.kdfparams.salt)
            .map_err(|e| WalletError::KeystoreParse(format!("salt hex: {}", e)))?;
        let iv = hex::decode(&crypto.cipherparams.iv)
            .map_err(|e| WalletError::KeystoreParse(format!("iv hex: {}", e)))?;
        let ciphertext = hex::decode(&crypto.ciphertext)
            .map_err(|e| WalletError::KeystoreParse(format!("ciphertext hex: {}", e)))?;
        let mac_expected = hex::decode(&crypto.mac)
            .map_err(|e| WalletError::KeystoreParse(format!("mac hex: {}", e)))?;

        if iv.len() != 16 {
            return Err(WalletError::KeystoreParse(format!(
                "iv must be 16 bytes, got {}",
                iv.len()
            )));
        }

        // PBA-L4-010: hard floor (refuse), then the modern floor (warn).
        let below_modern_kdf = check_kdf_cost(crypto.kdfparams.c)?;

        // 1. Derive 32-byte key via PBKDF2-HMAC-SHA256.
        // CP-B-005: the derived key, the decrypted plaintext and the hex round-
        // trip below are all secp256k1 key material; wrap each in `Zeroizing` so
        // the copies are wiped on drop rather than left in freed heap/stack for a
        // core dump / `/proc/<pid>/mem` reader to recover.
        let mut derived = Zeroizing::new([0u8; 32]);
        pbkdf2::pbkdf2_hmac::<sha2::Sha256>(
            passphrase.as_bytes(),
            &salt,
            crypto.kdfparams.c,
            &mut derived[..],
        );

        // 2. MAC check (keccak256 of dkey[16..32] || ciphertext).
        // SECREM-02 6.3 (CITRATE_COMPUTE_POOL-2026-05-31-007): the
        // comparison is constant-time (`subtle::ConstantTimeEq`) so
        // a byte-position timing oracle cannot speed up offline
        // passphrase guessing against a captured keystore.
        let mut hasher = Keccak256::new();
        hasher.update(&derived[16..32]);
        hasher.update(&ciphertext);
        let mac_actual = hasher.finalize();
        use subtle::ConstantTimeEq;
        if mac_actual
            .as_slice()
            .ct_eq(mac_expected.as_slice())
            .unwrap_u8()
            == 0
        {
            return Err(WalletError::InvalidPassphrase);
        }

        // 3. AES-128-CTR decrypt. Key = dkey[0..16].
        use aes::cipher::{KeyIvInit, StreamCipher};
        type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;
        let mut plaintext = Zeroizing::new(ciphertext.clone());
        let mut cipher = Aes128Ctr::new((&derived[0..16]).into(), iv.as_slice().into());
        cipher.apply_keystream(&mut plaintext[..]);

        if plaintext.len() != 32 {
            return Err(WalletError::KeystoreParse(format!(
                "decrypted key is {} bytes, expected 32",
                plaintext.len()
            )));
        }

        // 4. Construct wallet + verify keystore-declared address.
        let hex_key = Zeroizing::new(hex::encode(&plaintext[..]));
        let mut wallet = Self::from_hex(&hex_key)?;
        wallet.below_modern_kdf = below_modern_kdf;
        if let Some(declared) = file.address.as_deref() {
            let cleaned = declared.trim().trim_start_matches("0x");
            let declared_addr = H160::from_slice(
                &hex::decode(cleaned)
                    .map_err(|e| WalletError::KeystoreParse(format!("addr hex: {}", e)))?,
            );
            wallet.verify_address(declared_addr)?;
        }
        Ok(wallet)
        // END SHARED-KEYSTORE
    }

    /// The keystore's PBKDF2 iteration count if it is below the modern floor.
    pub fn below_modern_kdf(&self) -> Option<u32> {
        self.below_modern_kdf
    }

    // BEGIN SHARED-KEYSTORE (PBA-L4-010)
    /// Emit the warnings gathered while loading. Call once logging is
    /// initialised: the wallet is loaded before the tracing subscriber (and
    /// before any thread) exists, so nothing can be logged at load time.
    pub fn log_load_warnings(&self) {
        if let Some(iterations) = self.below_modern_kdf {
            tracing::warn!(
                iterations,
                floor = MIN_MODERN_PBKDF2_ITERS,
                "keystore uses a below-modern PBKDF2 iteration count; \
                 re-mint with scrypt + AES-256 at next key rotation \
                 (ENCRYPT-S1 WP-10)"
            );
        }
    }
    // END SHARED-KEYSTORE

    /// Build from a hex-encoded private key.
    pub fn from_hex(hex_key: &str) -> Result<Self, WalletError> {
        let trimmed = hex_key.trim().trim_start_matches("0x");
        if trimmed.len() != 64 {
            return Err(WalletError::BadHexLength(trimmed.len()));
        }
        // CP-B-005: the decoded key bytes are the raw private key; wipe them on
        // drop. `SigningKey` zeroizes its own copy, so only these intermediates
        // needed wrapping.
        let bytes =
            Zeroizing::new(hex::decode(trimmed).map_err(|e| WalletError::BadHex(e.to_string()))?);
        let mut sk_bytes = Zeroizing::new([0u8; 32]);
        sk_bytes.copy_from_slice(&bytes);
        let signing_key =
            SigningKey::from_bytes(&(*sk_bytes).into()).map_err(|_| WalletError::BadSecret)?;
        let address = address_from_signing_key(&signing_key);
        Ok(Self {
            signing_key,
            address,
            below_modern_kdf: None,
        })
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
    /// Recoverable secp256k1 signature over an arbitrary 32-byte digest,
    /// returned as the 65-byte `r || s || v` form (`v` is the recovery id, 0/1).
    ///
    /// "Recoverable" is the load-bearing word: a verifier can recover the signing
    /// ADDRESS from the signature and the digest alone. That is what lets the
    /// federated gather verify a contribution with **no roster and no distributed
    /// key registry** — the claimed identity IS the address, and a signature
    /// either recovers to it or does not.
    ///
    /// Unlike `trial_recovery_from_prehash`'s fallback in `sign_eip1559`, a
    /// failure here is returned rather than defaulted to recovery id 0: a wrong
    /// `v` yields a signature that recovers to the WRONG address, which the
    /// gather would read as a different node — or as forgery.
    pub fn sign_digest_recoverable(&self, digest: &[u8; 32]) -> Result<[u8; 65], WalletError> {
        let sig: Signature = self
            .signing_key
            .sign_prehash(digest)
            .map_err(|e| WalletError::Sign(format!("{}", e)))?;
        let recid = RecoveryId::trial_recovery_from_prehash(
            &self.signing_key.verifying_key().clone(),
            digest,
            &sig,
        )
        .map_err(|e| WalletError::Sign(format!("recovery id: {}", e)))?;

        let mut out = [0u8; 65];
        out[..64].copy_from_slice(&sig.to_bytes());
        out[64] = recid.to_byte();
        Ok(out)
    }

    /// Recover the signing address from a digest and a 65-byte recoverable
    /// signature. Pure — no key material, which is why a verifier can run it.
    pub fn recover_address(digest: &[u8; 32], sig65: &[u8]) -> Result<H160, WalletError> {
        if sig65.len() != 65 {
            return Err(WalletError::Sign(format!(
                "recoverable signature must be 65 bytes, got {}",
                sig65.len()
            )));
        }
        let sig = Signature::from_slice(&sig65[..64])
            .map_err(|e| WalletError::Sign(format!("bad r||s: {}", e)))?;
        let recid = RecoveryId::from_byte(sig65[64])
            .ok_or_else(|| WalletError::Sign(format!("bad recovery id {}", sig65[64])))?;
        let vk = k256::ecdsa::VerifyingKey::recover_from_prehash(digest, &sig, recid)
            .map_err(|e| WalletError::Sign(format!("recover: {}", e)))?;

        let point = vk.to_encoded_point(false);
        let h = keccak256(&point.as_bytes()[1..]);
        let mut addr = [0u8; 20];
        addr.copy_from_slice(&h[12..]);
        Ok(H160::from(addr))
    }

    pub fn sign_eip1559(&self, tx: &Eip1559Tx) -> Result<Vec<u8>, WalletError> {
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
                .unwrap_or_else(|_| RecoveryId::from_byte(0).expect("0 is a valid RecoveryId"));
                (bytes, rid.to_byte())
            })
            .map_err(|e| WalletError::Sign(format!("{}", e)))?;

        let r_bytes = &sig[..32];
        let s_bytes = &sig[32..64];
        let r = U256::from_big_endian(r_bytes);
        let s = U256::from_big_endian(s_bytes);
        let y_parity = recid as u64;

        Ok(encode_eip1559_with_signature(tx, y_parity, r, s))
    }
}

/// An EIP-1559 (type-2) transaction we can sign. Minimal surface:
/// no access list, no blob fields — the training-worker HTTP chain
/// client only needs simple value-carrying + calldata-bearing writes.
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

fn encode_eip1559_with_signature(tx: &Eip1559Tx, y_parity: u64, r: U256, s_: U256) -> Vec<u8> {
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
/// HTTP chain clients to surface the pre-broadcast hash.
pub fn tx_hash_of_signed(signed_rlp: &[u8]) -> H256 {
    H256::from(keccak256(signed_rlp))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Well-known secp256k1 key for deterministic tests.
    // Address: 0x7e5f4552091a69125d5dfcb7b8c2659029395bdf
    const TEST_HEX: &str = "0000000000000000000000000000000000000000000000000000000000000001";

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
        let err = Wallet::from_hex("abcd").expect_err("must reject");
        assert!(matches!(err, WalletError::BadHexLength(_)));
    }

    #[test]
    fn rejects_invalid_hex_chars() {
        let bad = "z".repeat(64);
        let err = Wallet::from_hex(&bad).expect_err("must reject");
        assert!(matches!(err, WalletError::BadHex(_)));
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
        let err = w.verify_address(declared).expect_err("must reject");
        assert!(matches!(err, WalletError::AddressMismatch { .. }));
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

    // ── Keystore (Web3 SSv3) tests ──────────────────────────────

    /// Build a keystore JSON payload encrypting TEST_HEX under the
    /// given passphrase. Mirrors the SDK's W-01 encryption pipeline
    /// so test fixtures are valid at both ends.
    fn build_keystore_json(passphrase: &str) -> String {
        build_keystore_json_with_cost(passphrase, MIN_PBKDF2_ITERS)
    }

    fn build_keystore_json_with_cost(passphrase: &str, c: u32) -> String {
        use aes::cipher::{KeyIvInit, StreamCipher};
        type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;

        // Deterministic salt + iv (fixed values for test repeatability).
        let salt = [0x11u8; 32];
        let iv = [0x22u8; 16];

        // Derive key.
        let mut derived = [0u8; 32];
        pbkdf2::pbkdf2_hmac::<sha2::Sha256>(passphrase.as_bytes(), &salt, c, &mut derived);

        // Encrypt plaintext private key.
        let plaintext = hex::decode(TEST_HEX).expect("decode plaintext");
        let mut ciphertext = plaintext.clone();
        let mut cipher = Aes128Ctr::new((&derived[0..16]).into(), (&iv).into());
        cipher.apply_keystream(&mut ciphertext);

        // MAC = keccak256(dkey[16..32] || ciphertext)
        let mut hasher = Keccak256::new();
        hasher.update(&derived[16..32]);
        hasher.update(&ciphertext);
        let mac = hasher.finalize();

        serde_json::json!({
            "version": 3,
            "address": "7e5f4552091a69125d5dfcb7b8c2659029395bdf",
            "crypto": {
                "cipher": "aes-128-ctr",
                "cipherparams": { "iv": hex::encode(iv) },
                "ciphertext": hex::encode(&ciphertext),
                "kdf": "pbkdf2",
                "kdfparams": {
                    "prf": "hmac-sha256",
                    "c": c,
                    "salt": hex::encode(salt),
                    "dklen": 32,
                },
                "mac": hex::encode(mac),
            },
        })
        .to_string()
    }

    fn write_keystore(passphrase: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let json = build_keystore_json(passphrase);
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        path.push(format!("citrate-training-keystore-{}-{}.json", pid, n));
        std::fs::write(&path, json).expect("write keystore");
        path
    }

    #[test]
    fn loads_from_keystore_with_correct_passphrase() {
        let path = write_keystore("hunter2");
        let w =
            Wallet::from_keystore(path.to_str().expect("path utf8"), "hunter2").expect("unlock");
        assert_eq!(
            w.address(),
            "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"
                .parse::<H160>()
                .expect("addr")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_wrong_passphrase() {
        let path = write_keystore("correct");
        let err = Wallet::from_keystore(path.to_str().expect("path utf8"), "wrong")
            .expect_err("must reject wrong passphrase");
        assert!(matches!(err, WalletError::InvalidPassphrase));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_missing_keystore_file() {
        let err = Wallet::from_keystore("/nonexistent/path.json", "x")
            .expect_err("must reject missing file");
        assert!(matches!(err, WalletError::KeystoreIo(_)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let tmp = std::env::temp_dir().join("bad-version-training.json");
        std::fs::write(
            &tmp,
            serde_json::json!({
                "version": 2,
                "crypto": {
                    "cipher": "aes-128-ctr",
                    "cipherparams": { "iv": "00".repeat(16) },
                    "ciphertext": "",
                    "kdf": "pbkdf2",
                    "kdfparams": {
                        "prf": "hmac-sha256",
                        "c": 1024,
                        "salt": "00".repeat(32),
                        "dklen": 32,
                    },
                    "mac": "00".repeat(32),
                }
            })
            .to_string(),
        )
        .expect("write bad fixture");
        let err = Wallet::from_keystore(tmp.to_str().expect("path utf8"), "x")
            .expect_err("must reject v2");
        assert!(matches!(err, WalletError::UnsupportedKeystoreVersion(2)));
        let _ = std::fs::remove_file(tmp);
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

    // ── PBA-L4-010 ──────────────────────────────────────────────────

    #[test]
    fn pba_l4_010_a_keystore_below_the_kdf_floor_is_refused() {
        for c in [1u32, 1024, MIN_PBKDF2_ITERS - 1] {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "citrate-weak-kdf-{}-{}.json",
                std::process::id(),
                c
            ));
            std::fs::write(&path, build_keystore_json_with_cost("pw", c)).expect("write");
            let err = Wallet::from_keystore(path.to_str().unwrap(), "pw")
                .expect_err("a keystore below the KDF floor must be refused");
            assert!(
                matches!(err, WalletError::WeakKdf { iterations, floor }
                    if iterations == c && floor == MIN_PBKDF2_ITERS),
                "c={c}: {err:?}"
            );
            let _ = std::fs::remove_file(path);
        }
        // At the floor it loads.
        let path = write_keystore("pw");
        assert!(Wallet::from_keystore(path.to_str().unwrap(), "pw").is_ok());
        let _ = std::fs::remove_file(path);
    }

    /// One test (not two) because both halves mutate the same process env vars.
    #[test]
    fn pba_l4_010_from_env_removes_the_secret_from_the_environment() {
        // Serialise with every other test that touches the process env.
        let _env = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        std::env::remove_var(ENV_KEYSTORE_PATH);
        std::env::set_var(ENV_PRIVATE_KEY_HEX, TEST_HEX);
        let w = Wallet::from_env().expect("load from raw hex env");
        assert_eq!(w.address(), Wallet::from_hex(TEST_HEX).unwrap().address());
        assert!(
            std::env::var_os(ENV_PRIVATE_KEY_HEX).is_none(),
            "the raw key must not stay in the environment after load"
        );

        let path = write_keystore("hunter2");
        std::env::set_var(ENV_KEYSTORE_PATH, &path);
        std::env::set_var(ENV_KEYSTORE_PASSPHRASE, "hunter2");
        std::env::set_var(ENV_PRIVATE_KEY_HEX, TEST_HEX);
        Wallet::from_env().expect("load from keystore env");
        assert!(
            std::env::var_os(ENV_KEYSTORE_PASSPHRASE).is_none(),
            "the passphrase must not stay in the environment after load"
        );
        assert!(
            std::env::var_os(ENV_PRIVATE_KEY_HEX).is_none(),
            "a raw key set alongside a keystore must be cleared too"
        );
        // Keystore path without a passphrase: an error, and a raw key set
        // alongside is still cleared.
        std::env::remove_var(ENV_KEYSTORE_PASSPHRASE);
        std::env::set_var(ENV_PRIVATE_KEY_HEX, TEST_HEX);
        assert!(matches!(
            Wallet::from_env(),
            Err(WalletError::MissingPassphrase)
        ));
        assert!(
            std::env::var_os(ENV_PRIVATE_KEY_HEX).is_none(),
            "a raw key must be cleared even when the keystore load fails"
        );
        std::env::remove_var(ENV_KEYSTORE_PATH);
        let _ = std::fs::remove_file(path);
        assert!(matches!(Wallet::from_env(), Err(WalletError::NoKeySource)));
    }

    /// PBA-L4-010: the two wallet copies diverged once (only one warned about a
    /// weak KDF). Until they are unified into one crate, the keystore and
    /// env-secret code must be byte-identical between them.
    #[test]
    fn the_two_wallet_copies_share_one_keystore_implementation() {
        fn shared(src: &str) -> String {
            let mut out = String::new();
            let mut on = false;
            for line in src.lines() {
                if line.trim_start().starts_with("// BEGIN SHARED-KEYSTORE") {
                    on = true;
                }
                if on {
                    out.push_str(line);
                    out.push('\n');
                }
                if line.trim_start().starts_with("// END SHARED-KEYSTORE") {
                    on = false;
                }
            }
            out
        }
        let here = shared(include_str!("wallet.rs"));
        let there = shared(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pool-coordinator/src/wallet.rs"
        )));
        assert!(here.matches("BEGIN SHARED-KEYSTORE").count() >= 3);
        assert_eq!(
            here, there,
            "wallet.rs copies diverged inside SHARED-KEYSTORE"
        );
    }

    #[test]
    fn a_below_modern_keystore_is_reported_after_load() {
        assert_eq!(
            check_kdf_cost(MIN_PBKDF2_ITERS).ok(),
            Some(Some(MIN_PBKDF2_ITERS))
        );
        assert_eq!(
            check_kdf_cost(MIN_MODERN_PBKDF2_ITERS - 1).ok(),
            Some(Some(MIN_MODERN_PBKDF2_ITERS - 1))
        );
        assert_eq!(check_kdf_cost(MIN_MODERN_PBKDF2_ITERS).ok(), Some(None));
        let path = write_keystore("pw");
        let w = Wallet::from_keystore(path.to_str().unwrap(), "pw").expect("load");
        assert_eq!(w.below_modern_kdf(), Some(MIN_PBKDF2_ITERS));
        let out = logged(|| w.log_load_warnings());
        assert!(
            out.contains("below-modern PBKDF2"),
            "warning not logged: {out:?}"
        );
        assert!(
            out.contains(&format!("iterations={MIN_PBKDF2_ITERS}")),
            "{out:?}"
        );
        let clean = Wallet::from_hex(TEST_HEX).unwrap();
        assert_eq!(clean.below_modern_kdf(), None);
        assert_eq!(
            logged(|| clean.log_load_warnings()),
            "",
            "a clean key logs nothing"
        );
        let _ = std::fs::remove_file(path);
    }

    /// Run `f` under a tracing subscriber that captures its output.
    fn logged(f: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Captured {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let cap = Captured::default();
        let sink = cap.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || sink.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(sub, f);
        let bytes = cap.0.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }
}
