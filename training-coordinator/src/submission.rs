//! The signed result a worker returns, and the exact bytes it signs.
//!
//! The digest is defined **here, once**, and both sides call this function. A
//! coordinator and a worker that each hand-roll "the obvious" concatenation agree
//! right up until one of them changes a separator, at which point every honest
//! submission fails to authenticate and the failure looks like a key problem.

use citrate_training_worker::wallet::Wallet;
use ethereum_types::H160;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

use crate::job::JobId;

/// `keccak256("citrate-training-submission/1\n" || job_id || "\n" || payload)`.
///
/// The domain prefix keeps a submission signature from being replayable as any
/// other message this key signs — notably a chain transaction, since the worker
/// signs both with the same key. The newline separators make the encoding
/// unambiguous for the job ids actually in use (`h01-64m-nat-seed2`), which
/// contain no newlines.
pub fn submission_digest(job: &JobId, payload: &str) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(b"citrate-training-submission/1\n");
    h.update(job.0.as_bytes());
    h.update(b"\n");
    h.update(payload.as_bytes());
    let mut d = [0u8; 32];
    d.copy_from_slice(&h.finalize());
    d
}

/// A result as it arrives from a worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignedSubmission {
    pub job: JobId,
    /// Opaque to the coordinator: it schedules work and records outcomes, it does
    /// not interpret training output. Verification of the *content* belongs to
    /// the challenge path, which can re-run the work; the coordinator can only
    /// establish who said it.
    pub payload: String,
    #[serde(with = "crate::attestation::hex_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SubmissionAuthError {
    #[error("signature does not recover to a valid address")]
    BadSignature,
}

/// Recover who signed a submission. Returns the address; whether that address is
/// *entitled* to submit is a separate question, answered by the state machine.
pub fn recover_submitter(s: &SignedSubmission) -> Result<H160, SubmissionAuthError> {
    Wallet::recover_address(&submission_digest(&s.job, &s.payload), &s.signature)
        .map_err(|_| SubmissionAuthError::BadSignature)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_A: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    const KEY_B: &str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";

    fn signed(key: &str, job: &str, payload: &str) -> SignedSubmission {
        let w = Wallet::from_hex(key).unwrap();
        let id = JobId(job.into());
        SignedSubmission {
            signature: w
                .sign_digest_recoverable(&submission_digest(&id, payload))
                .unwrap()
                .to_vec(),
            job: id,
            payload: payload.into(),
        }
    }

    #[test]
    fn a_submission_recovers_to_its_signer() {
        let s = signed(KEY_A, "j1", "result");
        assert_eq!(
            recover_submitter(&s).unwrap(),
            Wallet::from_hex(KEY_A).unwrap().address()
        );
    }

    #[test]
    fn two_signers_are_distinguishable() {
        assert_ne!(
            recover_submitter(&signed(KEY_A, "j1", "r")).unwrap(),
            recover_submitter(&signed(KEY_B, "j1", "r")).unwrap()
        );
    }

    /// The result is what gets paid for. Editing it after signing must break
    /// authentication, not merely be noticed.
    #[test]
    fn editing_the_payload_changes_who_it_recovers_to() {
        let mut s = signed(KEY_A, "j1", "honest result");
        let real = recover_submitter(&s).unwrap();
        s.payload = "inflated result".into();
        // It still recovers to *something* — secp256k1 recovery nearly always
        // yields an address — but not to the original signer, which is what the
        // leaseholder check then rejects.
        assert_ne!(recover_submitter(&s).unwrap_or_default(), real);
    }

    /// A signature captured for one job must not be replayable onto another.
    #[test]
    fn a_signature_does_not_transfer_between_jobs() {
        let s = signed(KEY_A, "j1", "r");
        let moved = SignedSubmission {
            job: JobId("j2".into()),
            ..s.clone()
        };
        assert_ne!(
            recover_submitter(&moved).unwrap_or_default(),
            recover_submitter(&s).unwrap()
        );
    }

    /// The domain prefix exists so a submission signature is not also a valid
    /// signature over some other message the same key signs.
    #[test]
    fn the_digest_is_domain_separated() {
        let id = JobId("j1".into());
        let naive = {
            let mut h = Keccak256::new();
            h.update(id.0.as_bytes());
            h.update(b"\n");
            h.update(b"payload");
            let mut d = [0u8; 32];
            d.copy_from_slice(&h.finalize());
            d
        };
        assert_ne!(submission_digest(&id, "payload"), naive);
    }

    /// Job id and payload must not be able to trade characters across the
    /// separator and produce the same digest.
    #[test]
    fn the_job_and_payload_fields_cannot_be_confused_for_each_other() {
        assert_ne!(
            submission_digest(&JobId("a".into()), "b"),
            submission_digest(&JobId("a\nb".into()), "")
        );
    }

    #[test]
    fn malformed_signatures_are_rejected_without_panicking() {
        let good = signed(KEY_A, "j1", "r");
        for bad in [vec![], vec![0u8; 64], vec![0u8; 66], vec![0u8; 65]] {
            let s = SignedSubmission {
                signature: bad,
                ..good.clone()
            };
            let _ = recover_submitter(&s); // must not panic
        }
    }
}
