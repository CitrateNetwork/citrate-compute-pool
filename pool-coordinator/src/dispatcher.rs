//! Stateless round-robin member selection.
//!
//! Per the planset (CM-05 sprint card "Notes" section): the
//! coordinator picks `members[hash(jobId) % memberCount]` rather than
//! holding a shared cursor. Two consequences worth being explicit
//! about:
//!
//! 1. **No shared state across coordinator restarts.** A new
//!    coordinator binary started mid-stream produces the SAME member
//!    pick for any given jobId.
//! 2. **No replication problem.** Multiple members running the
//!    coordinator binary all pick the same target; if more than one
//!    is the elected coordinator (e.g. epoch-boundary race), they
//!    both call recordDispatch and the contract's "Job not pending"
//!    guard catches the second one.
//!
//! Hash function: keccak256 of the job_id (big-endian u64 → 32 bytes).
//! Output mod `member_count`.

use ethereum_types::H160;
use sha3::{Digest, Keccak256};

/// Newtype wrapping a member's address. Helps the type system catch
/// "did I pass the wallet address or the member address?" mistakes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemberId(pub H160);

/// Pick a member for a given job_id from a set of active members.
/// Returns `None` if the member set is empty.
pub fn select_member(job_id: u64, members: &[MemberId]) -> Option<MemberId> {
    if members.is_empty() {
        return None;
    }
    let mut hasher = Keccak256::new();
    hasher.update(job_id.to_be_bytes());
    let h = hasher.finalize();
    // Use the last 8 bytes as a u64 to avoid bias from naive byte
    // truncation (any 8 bytes are statistically uniform over keccak
    // output, but tail bytes are slightly more conventional).
    let mut tail = [0u8; 8];
    tail.copy_from_slice(&h[24..32]);
    let n = u64::from_be_bytes(tail);
    let idx = (n % members.len() as u64) as usize;
    Some(members[idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(byte: u8) -> MemberId {
        MemberId(H160::from([byte; 20]))
    }

    #[test]
    fn deterministic_per_job_id() {
        let ms = [member(1), member(2), member(3)];
        for j in [0u64, 1, 42, 1_000_000, u64::MAX] {
            let a = select_member(j, &ms).expect("non-empty");
            let b = select_member(j, &ms).expect("non-empty");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn different_job_ids_can_yield_different_members() {
        let ms = [member(1), member(2), member(3)];
        // 100 distinct job_ids should hit at least 2 distinct members.
        let picks: std::collections::HashSet<MemberId> = (0..100u64)
            .map(|j| select_member(j, &ms).expect("non-empty"))
            .collect();
        assert!(picks.len() >= 2, "no rotation across 100 job ids");
    }

    #[test]
    fn empty_pool_returns_none() {
        let ms: Vec<MemberId> = vec![];
        assert!(select_member(0, &ms).is_none());
    }

    #[test]
    fn single_member_always_returned() {
        let ms = [member(0xab)];
        for j in 0..50u64 {
            let m = select_member(j, &ms).expect("non-empty");
            assert_eq!(m, ms[0]);
        }
    }
}
