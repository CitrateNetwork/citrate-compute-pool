//! Merkle epoch root construction per ADR-008.
//!
//! Leaves are built from (epoch, step, worker, step_commit,
//! prev_weights_hash) tuples, sorted canonically by (step, worker)
//! so the root is deterministic given the set of commits. Tree is
//! binary, internal nodes are `keccak256(sorted_pair)`, zero-
//! padding to the next power of two.
//!
//! The sort-before-hash convention matches the contract's verifier
//! (ComputePoolTraining._verifyMerkleProof), which takes the
//! smaller-of-the-two-as-left convention at every level.

use sha3::{Digest, Keccak256};

use crate::types::{B256, StepCommit};

/// Compute the Merkle leaf for a step commit per ADR-008:
///
///   leaf = keccak256(abi.encode(
///       uint32 epoch,
///       uint32 step,
///       address worker,
///       bytes32 step_commit,
///       bytes32 prev_weights_hash
///   ))
///
/// Solidity `abi.encode` uses 32-byte-per-field padded layout.
/// Each uint32 is left-padded with 28 zero bytes; address is left-
/// padded with 12 zero bytes.
pub fn compute_leaf(commit: &StepCommit) -> B256 {
    let mut hasher = Keccak256::new();

    // uint32 epoch, padded to 32 bytes big-endian
    let mut buf = [0u8; 32];
    buf[28..32].copy_from_slice(&commit.epoch.to_be_bytes());
    hasher.update(buf);

    // uint32 step
    let mut buf = [0u8; 32];
    buf[28..32].copy_from_slice(&commit.step.to_be_bytes());
    hasher.update(buf);

    // address worker (20 bytes), padded to 32 with 12 leading zeros
    let mut buf = [0u8; 32];
    buf[12..32].copy_from_slice(commit.worker.as_bytes());
    hasher.update(buf);

    // bytes32 step_commit
    hasher.update(commit.commitment.as_bytes());

    // bytes32 prev_weights_hash
    hasher.update(commit.prev_weights.as_bytes());

    let out = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    B256::from(h)
}

/// Hash two sibling nodes with the sorted-pair convention used by
/// the contract's Merkle verifier: the smaller-valued child goes
/// left.
///
/// RM-J3 (post-RM-I-3): mirrors the SOL-20 domain-separator scheme
/// the contract's `_verifyMerkleProof` uses. Internal nodes are
/// prefixed with `MERKLE_INTERNAL_PREFIX = 0x01` to prevent the
/// second-preimage attack where an internal hash from a larger tree
/// could be presented as a leaf in a smaller tree. WP-I1.8 added the
/// prefix to the mock's verifier (`training-worker/src/chain.rs::
/// verify_merkle_proof`); without it here too, the worker builds
/// proofs against an unprefixed root and they fail verification both
/// in the mock and on-chain. The prefix matches
/// `contracts/src/ComputePoolTraining.sol:761-762` exactly.
const MERKLE_LEAF_PREFIX: u8 = 0x00;
const MERKLE_INTERNAL_PREFIX: u8 = 0x01;

fn hash_pair(a: B256, b: B256) -> B256 {
    let (left, right) = if a.as_bytes() <= b.as_bytes() {
        (a, b)
    } else {
        (b, a)
    };
    let mut hasher = Keccak256::new();
    hasher.update([MERKLE_INTERNAL_PREFIX]);
    hasher.update(left.as_bytes());
    hasher.update(right.as_bytes());
    let out = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    B256::from(h)
}

/// Promote a raw leaf into the leaf domain via `keccak(0x00 || leaf)`.
/// Mirrors the first line of `_verifyMerkleProof` on-chain. Used at the
/// bottom of `compute_epoch_root` so the root we build matches the
/// root the contract reconstructs during verification.
fn promote_leaf(leaf: B256) -> B256 {
    let mut hasher = Keccak256::new();
    hasher.update([MERKLE_LEAF_PREFIX]);
    hasher.update(leaf.as_bytes());
    let out = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    B256::from(h)
}

/// Build the epoch Merkle root from a set of step commits. Sorts
/// canonically by (step, worker) first, then computes leaves, then
/// reduces bottom-up with zero-padding.
///
/// Returns the root and the ordered leaf vector (useful for
/// subsequent proof construction).
pub fn compute_epoch_root(commits: &[StepCommit]) -> (B256, Vec<B256>) {
    let mut sorted: Vec<&StepCommit> = commits.iter().collect();
    sorted.sort_by(|a, b| a.step.cmp(&b.step).then_with(|| a.worker.cmp(&b.worker)));

    // RM-J3: returned `leaves` are the UNPREFIXED leaves — the values
    // the contract's `_verifyMerkleProof(proof, root, leaf)` expects as
    // the `leaf` argument. They are what `challenge_step` and
    // `compute_proof` consume. The PROMOTED form (with `0x00` prefix)
    // is used only internally during root reduction.
    let leaves: Vec<B256> = sorted.iter().map(|c| compute_leaf(c)).collect();

    if leaves.is_empty() {
        return (B256::zero(), leaves);
    }
    if leaves.len() == 1 {
        // Single-leaf root: contract's verifier with empty proof
        // produces `keccak(0x00 || leaf)`. Match that.
        return (promote_leaf(leaves[0]), leaves);
    }

    // Bottom-up Merkle reduction. Promote leaves into the leaf domain
    // first (`keccak(0x00 || leaf)`), then reduce with `hash_pair`
    // which adds `0x01` prefix on each internal node.
    let mut level: Vec<B256> = leaves.iter().map(|l| promote_leaf(*l)).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() {
                level[i + 1]
            } else {
                B256::zero()
            };
            next.push(hash_pair(left, right));
            i += 2;
        }
        level = next;
    }
    (level[0], leaves)
}

/// Generate a Merkle proof for a specific leaf index in an already-
/// built tree (siblings from leaf to root). Used by challengers
/// constructing off-chain proofs for `challengeStep`.
pub fn compute_proof(leaves: &[B256], target: usize) -> Vec<B256> {
    let mut proof = Vec::new();
    if leaves.is_empty() || target >= leaves.len() {
        return proof;
    }
    if leaves.len() == 1 {
        // Single-leaf tree: the contract's verifier with empty proof
        // produces `keccak(0x00 || leaf)`, which equals the root we
        // returned from `compute_epoch_root`. Empty proof verifies
        // correctly. RM-J3 (post-RM-I-3 prefix discipline).
        return proof;
    }

    // RM-J3: walk the tree at the SAME level shape `compute_epoch_root`
    // uses — start at the promoted-leaf level (`keccak(0x00 || leaf)`),
    // then reduce with `hash_pair` (adds `0x01` per level). The proof's
    // first sibling is therefore a promoted leaf; the contract's
    // verifier — given the unprefixed `leaf` — will compute
    // `keccak(0x00 || leaf)` for the starting `computed`, then combine
    // with our first proof element using `keccak(0x01 || ...)`. That
    // matches the level shape exactly.
    let mut level: Vec<B256> = leaves.iter().map(|l| promote_leaf(*l)).collect();
    let mut idx = target;
    while level.len() > 1 {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        let sibling = if sibling_idx < level.len() {
            level[sibling_idx]
        } else {
            B256::zero()
        };
        proof.push(sibling);

        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i < level.len() {
            let left = level[i];
            let right = if i + 1 < level.len() {
                level[i + 1]
            } else {
                B256::zero()
            };
            next.push(hash_pair(left, right));
            i += 2;
        }
        level = next;
        idx /= 2;
    }
    proof
}

/// Verify a Merkle proof — the mirror of the on-chain verifier in
/// `ComputePoolTraining._verifyMerkleProof`. Used by worker-side
/// unit tests to assert round-trip correctness before shipping
/// proofs on-chain.
///
/// RM-J3 (post-RM-I-3): promotes the raw `leaf` into the leaf domain
/// (`keccak(0x00 || leaf)`) before walking the proof, mirroring the
/// contract's behaviour. Internal-node hashing (`hash_pair`) adds the
/// `0x01` prefix automatically.
pub fn verify_proof(proof: &[B256], root: B256, leaf: B256) -> bool {
    let mut computed = promote_leaf(leaf);
    for sibling in proof {
        computed = hash_pair(computed, *sibling);
    }
    computed == root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepIndex, WorkerAddress};
    use ethereum_types::Address;

    fn addr(b: u8) -> Address {
        Address::repeat_byte(b)
    }

    fn mk_commit(step: StepIndex, worker: WorkerAddress, byte: u8) -> StepCommit {
        StepCommit {
            epoch: 0,
            step,
            worker,
            commitment: B256::repeat_byte(byte),
            prev_weights: B256::repeat_byte(0xAA),
        }
    }

    #[test]
    fn empty_commits_yield_zero_root() {
        let (root, leaves) = compute_epoch_root(&[]);
        assert_eq!(root, B256::zero());
        assert!(leaves.is_empty());
    }

    #[test]
    fn single_commit_is_promoted_leaf_root() {
        // RM-J3: with the SOL-20 prefix scheme, a single-leaf root is
        // `keccak(0x00 || leaf)`, NOT the bare leaf. The contract's
        // verifier with empty proof produces this same value, so an
        // empty proof against this root and the unprefixed leaf
        // verifies correctly.
        let c = mk_commit(0, addr(1), 0x11);
        let (root, leaves) = compute_epoch_root(&[c.clone()]);
        assert_eq!(leaves.len(), 1);
        assert_eq!(root, promote_leaf(compute_leaf(&c)));
        // Round-trip check: empty proof + unprefixed leaf verifies.
        assert!(verify_proof(&[], root, compute_leaf(&c)));
    }

    #[test]
    fn root_deterministic_under_input_order() {
        let a = mk_commit(0, addr(1), 0x11);
        let b = mk_commit(0, addr(2), 0x22);
        let c = mk_commit(1, addr(1), 0x33);
        let (root_ab_c, _) = compute_epoch_root(&[a.clone(), b.clone(), c.clone()]);
        let (root_c_b_a, _) = compute_epoch_root(&[c, b, a]);
        assert_eq!(root_ab_c, root_c_b_a, "sort canonicalizes input order");
    }

    #[test]
    fn proof_round_trips_for_every_leaf() {
        let commits: Vec<StepCommit> = (0..4)
            .map(|i| mk_commit((i / 2) as u32, addr(i as u8 + 1), i as u8))
            .collect();
        let (root, leaves) = compute_epoch_root(&commits);
        for i in 0..leaves.len() {
            let proof = compute_proof(&leaves, i);
            assert!(verify_proof(&proof, root, leaves[i]), "leaf {} failed", i);
        }
    }

    #[test]
    fn proof_rejects_tampered_leaf() {
        let commits: Vec<StepCommit> = (0..3)
            .map(|i| mk_commit(i, addr(1), i as u8))
            .collect();
        let (root, leaves) = compute_epoch_root(&commits);
        let proof = compute_proof(&leaves, 0);
        let tampered = B256::repeat_byte(0xFF);
        assert!(!verify_proof(&proof, root, tampered));
    }

    #[test]
    fn changing_commitment_byte_changes_root() {
        let a = mk_commit(0, addr(1), 0x11);
        let a2 = mk_commit(0, addr(1), 0x12);
        let (r1, _) = compute_epoch_root(&[a]);
        let (r2, _) = compute_epoch_root(&[a2]);
        assert_ne!(r1, r2);
    }
}
