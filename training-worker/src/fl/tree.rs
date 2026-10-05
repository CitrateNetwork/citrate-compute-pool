//! The positional Merkle tree every FL round root uses (FL_ROUND_V1 §4).
//!
//! * leaf `i` = `keccak256(0x00 ‖ be32(i) ‖ payload_i)`: the index is inside the
//!   leaf, so a proof for one position can never be replayed at another;
//! * the leaf list is padded with zero words up to the next power of two;
//! * node = `keccak256(0x01 ‖ left ‖ right)`: the tag keeps a node from ever
//!   being mistaken for a leaf.
//!
//! `FederatedRoundLedger._verify` is the on-chain verifier of exactly this tree,
//! and the chain's replay tool computes it independently.

use super::{keccak, B32};

const LEAF_TAG: [u8; 1] = [0x00];
const NODE_TAG: [u8; 1] = [0x01];

/// Most leaves a tree may have (the on-chain record stores counts as u32).
pub const MAX_LEAVES: usize = 1 << 31;

pub fn leaf(index: u32, payload: &B32) -> B32 {
    keccak(&[&LEAF_TAG, &index.to_be_bytes(), payload])
}

pub fn node(left: &B32, right: &B32) -> B32 {
    keccak(&[&NODE_TAG, left, right])
}

fn leaves(payloads: &[B32]) -> Result<Vec<B32>, TreeError> {
    if payloads.is_empty() {
        return Err(TreeError::Empty);
    }
    if payloads.len() > MAX_LEAVES {
        return Err(TreeError::TooMany(payloads.len()));
    }
    let width = payloads.len().next_power_of_two();
    let mut level: Vec<B32> = payloads
        .iter()
        .enumerate()
        .map(|(i, p)| leaf(i as u32, p))
        .collect();
    level.resize(width, [0u8; 32]);
    Ok(level)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TreeError {
    #[error("a tree needs at least one leaf")]
    Empty,
    #[error("{0} leaves is more than a round may commit")]
    TooMany(usize),
    #[error("index {index} is outside a tree of {len} leaves")]
    Index { index: usize, len: usize },
}

/// The root over `payloads`, in order.
pub fn root(payloads: &[B32]) -> Result<B32, TreeError> {
    let mut level = leaves(payloads)?;
    while level.len() > 1 {
        level = level
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[l, r]| node(l, r))
            .collect();
    }
    Ok(level[0])
}

/// The sibling path for `index`, leaf level first.
pub fn proof(payloads: &[B32], index: usize) -> Result<Vec<B32>, TreeError> {
    if index >= payloads.len() {
        return Err(TreeError::Index {
            index,
            len: payloads.len(),
        });
    }
    let mut level = leaves(payloads)?;
    let mut i = index;
    let mut path = Vec::new();
    while level.len() > 1 {
        path.push(level[i ^ 1]);
        level = level
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[l, r]| node(l, r))
            .collect();
        i >>= 1;
    }
    Ok(path)
}

/// Check a sibling path. Mirrors the contract: the bit of `index` at each level
/// says which side the running hash sits on.
pub fn verify(root: &B32, index: u32, payload: &B32, path: &[B32]) -> bool {
    // No range check is needed: the index is inside the leaf preimage, so a path verified at
    // one index can never verify at another.
    let mut h = leaf(index, payload);
    let mut i = index;
    for sib in path {
        h = if i & 1 == 0 {
            node(&h, sib)
        } else {
            node(sib, &h)
        };
        i >>= 1;
    }
    &h == root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(n: u8) -> B32 {
        [n; 32]
    }

    #[test]
    fn a_single_leaf_root_is_that_leaf() {
        assert_eq!(root(&[p(7)]).expect("root"), leaf(0, &p(7)));
        assert!(verify(&leaf(0, &p(7)), 0, &p(7), &[]));
    }

    #[test]
    fn every_proof_verifies_and_only_at_its_own_index() {
        for n in 1..=9usize {
            let ps: Vec<B32> = (0..n as u8).map(p).collect();
            let r = root(&ps).expect("root");
            for i in 0..n {
                let path = proof(&ps, i).expect("proof");
                assert!(verify(&r, i as u32, &ps[i], &path), "n={n} i={i}");
                // Same payload, wrong index.
                let j = (i + 1) % n;
                if ps[j] != ps[i] {
                    assert!(!verify(&r, j as u32, &ps[i], &path), "n={n} i={i}");
                }
                // Index beyond the proof's height.
                assert!(!verify(&r, (i as u32) | (1 << path.len()), &ps[i], &path));
            }
        }
    }

    #[test]
    fn the_padding_shape_is_pinned() {
        // Three leaves pad to four: root = node(node(l0,l1), node(l2, 0)).
        let ps = [p(1), p(2), p(3)];
        let expect = node(
            &node(&leaf(0, &p(1)), &leaf(1, &p(2))),
            &node(&leaf(2, &p(3)), &[0u8; 32]),
        );
        assert_eq!(root(&ps).expect("root"), expect);
    }

    #[test]
    fn changing_any_payload_changes_the_root() {
        let ps: Vec<B32> = (0..5u8).map(p).collect();
        let r = root(&ps).expect("root");
        for i in 0..ps.len() {
            let mut q = ps.clone();
            q[i][0] ^= 1;
            assert_ne!(root(&q).expect("root"), r);
        }
    }

    #[test]
    fn a_leaf_is_not_a_node() {
        // Without tags, leaf(i, x) could collide with node(a, b) for crafted
        // inputs; the tags make the preimages structurally different.
        assert_ne!(leaf(0, &p(1)), node(&p(0), &p(1)));
    }

    #[test]
    fn a_long_hostile_path_is_just_false() {
        let ps = [p(1), p(2)];
        let r = root(&ps).expect("root");
        assert!(!verify(&r, 0, &p(1), &[[0u8; 32]; 70]));
        assert!(!verify(&r, u32::MAX, &p(1), &[]));
    }

    #[test]
    fn empty_and_out_of_range_are_errors() {
        assert_eq!(root(&[]), Err(TreeError::Empty));
        assert!(matches!(proof(&[p(1)], 1), Err(TreeError::Index { .. })));
    }
}
