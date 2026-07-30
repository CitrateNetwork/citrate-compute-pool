//! The real signer for the federated gather — replacing NAT's `ToyKeyedSigner`.
//!
//! `nat_federated::ToyKeyedSigner` computes `sig = H(key || msg || key)` and says
//! of itself: *"TEST STAND-IN — not for production"*. It has no public-key
//! separation, so verifying requires the verifier to hold every node's secret. In
//! practice that means shipping a roster of secrets to whoever verifies, which is
//! not a thing to do.
//!
//! This replaces it with the wallet the worker already loads.
//!
//! ## Why no new dependency and no new key story
//!
//! [`crate::wallet::Wallet`] already loads an **encrypted keystore**
//! (`CITRATE_TRAINING_KEYSTORE_PATH` + passphrase) and signs with secp256k1. It
//! is the same key the worker uses to send its `commitEpoch` transactions. Using
//! it here means a node has exactly one identity and one key custody story, not
//! two — and no second audited signer to keep in sync.
//!
//! ## Recoverable signatures remove the roster entirely
//!
//! secp256k1 signatures here are **recoverable**: the verifier recovers the
//! signing address from the digest and the signature alone. So `node_id` is the
//! address, and verification is "does this recover to the identity it claims?"
//!
//! That is strictly better than what it replaces. `ToyRosterVerifier` needs a
//! trusted `node_id → key` map distributed to every verifier and kept current as
//! nodes join and leave; a stale roster fails closed on an honest node.
//! [`RecoveringVerifier`] holds **no state at all** — there is nothing to
//! distribute and nothing to go stale.
//!
//! ## Scope: the LOCAL half
//!
//! This is the local-keystore signer. The remote (AWS-KMS) signer in
//! `citrate-inference-gateway` is the other half, and it is now unblocked —
//! NAT ADR-0011 made `Signer::sign` fallible precisely so a network signer has
//! somewhere to report a timeout. It is not wired here because a KMS call from
//! inside a synchronous trait method still needs a deliberate blocking strategy,
//! and the local path needs none: signing is in-process elliptic curve
//! arithmetic, and the only realistic failure is a key that never loaded.

use nat_federated::{SignError, Signer, Verifier};

use crate::wallet::Wallet;

/// Keccak-256, matching the digest convention the wallet's other signing path
/// uses and the one an on-chain verifier can reproduce.
fn digest_of(msg: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut h = Keccak256::new();
    h.update(msg);
    let out = h.finalize();
    let mut d = [0u8; 32];
    d.copy_from_slice(&out);
    d
}

/// Lowercase `0x`-hex of an address — the canonical `node_id` form.
///
/// Lowercased deliberately: a mixed-case (EIP-55) `node_id` and a lowercase one
/// would be different strings for the same node, and `node_id` is compared as a
/// string in the gather and used as a settlement key.
fn node_id_of(addr: &ethereum_types::H160) -> String {
    format!("0x{}", hex::encode(addr.as_bytes()))
}

/// A [`Signer`] backed by the worker's encrypted-keystore wallet.
pub struct WalletSigner {
    wallet: Wallet,
    node_id: String,
}

impl WalletSigner {
    pub fn new(wallet: Wallet) -> Self {
        let node_id = node_id_of(&wallet.address());
        Self { wallet, node_id }
    }

    /// Load from the same env the worker already uses, so the signing identity
    /// and the transacting identity cannot drift apart.
    pub fn from_env() -> Result<Self, crate::wallet::WalletError> {
        Ok(Self::new(Wallet::from_env()?))
    }
}

impl Signer for WalletSigner {
    fn node_id(&self) -> &str {
        &self.node_id
    }

    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, SignError> {
        self.wallet
            .sign_digest_recoverable(&digest_of(msg))
            .map(|s| s.to_vec())
            .map_err(|e| SignError(e.to_string()))
    }
}

/// A [`Verifier`] that recovers the signer's address and checks it against the
/// claimed `node_id`.
///
/// Stateless by construction — no roster, nothing to distribute, nothing to go
/// stale. A node that is not who it says it is fails because the signature
/// recovers to a different address, not because a list did or did not have it.
#[derive(Debug, Default, Clone, Copy)]
pub struct RecoveringVerifier;

impl Verifier for RecoveringVerifier {
    fn verify(&self, node_id: &str, msg: &[u8], sig: &[u8]) -> bool {
        // Every failure below is a `false`, never a panic: this runs over
        // untrusted input from the network, and a malformed signature is an
        // ordinary rejection rather than an exceptional condition.
        let Ok(recovered) = Wallet::recover_address(&digest_of(msg), sig) else {
            return false;
        };
        node_id_of(&recovered).eq_ignore_ascii_case(node_id)
    }
}

#[cfg(test)]
mod tests {
    include!("federated_signer_tests.rs");
}
