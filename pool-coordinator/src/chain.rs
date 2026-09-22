//! Chain-side abstraction.
//!
//! The daemon talks to the chain through this trait so the event-loop
//! logic is testable without a live RPC. The HTTP implementation
//! (`HttpChainAdapter`) lands in slice 2 alongside the real
//! `eth_subscribe newHeads` event subscription. Slice 1 ships only
//! the trait + types so the daemon's decision logic can be developed
//! and tested in isolation.

use async_trait::async_trait;
use ethereum_types::{H160, H256, U256};

use crate::error::CoordinatorError;

/// `ComputeRequested` event payload as the daemon sees it.
///
/// Decoded from the on-chain event:
///   ComputeRequested(uint256 indexed poolId, uint256 indexed jobId,
///                    address indexed requester, uint256 payment)
/// + the daemon decodes the job's `PoolJobSpec` from the on-chain
///   storage to recover prompt + max_tokens (the event itself only
///   carries IDs + payment to keep gas costs bounded).
///
/// `tx_hash`, `log_index`, and `block_number` are populated from
/// the on-chain log envelope and used by the event loop for
/// deduplication across overlapping polling windows (reorg
/// tolerance).
#[derive(Debug, Clone)]
pub struct ComputeRequestedEvent {
    pub pool_id: u64,
    pub job_id: u64,
    pub requester: H160,
    pub payment_grains: U256,
    pub prompt: String,
    pub max_tokens: u32,
    pub tx_hash: H256,
    pub log_index: u32,
    pub block_number: u64,
}

/// Subset of `PoolMember` fields the daemon needs for member-pool
/// reads. Mirrors `getMember` storage but flattened.
#[derive(Debug, Clone)]
pub struct PoolMemberInfo {
    pub address: H160,
    pub gpu_count: u32,
    /// Inactive members are skipped by the dispatcher.
    pub active: bool,
}

/// Outcome of a `recordDispatch` tx submission.
#[derive(Debug, Clone)]
pub enum RecordDispatchOutcome {
    /// Tx mined successfully.
    Confirmed { tx_hash: H256, block_number: u64 },
}

/// Trait the daemon's decision loop calls. Two implementations:
///
/// - `HttpChainAdapter` (slice 2) — real JSON-RPC client.
/// - `MockChain` (test only) — drives integration tests without a
///   live node.
#[async_trait]
pub trait ChainAdapter: Send + Sync {
    /// Address this daemon's wallet signs as. Equality with
    /// `coordinator_for(pool, epoch)` decides whether we act on an
    /// incoming event.
    fn self_address(&self) -> H160;

    /// Read `ComputePool.coordinatorFor(poolId, epoch)`.
    async fn coordinator_for(&self, pool_id: u64, epoch: u64) -> Result<H160, CoordinatorError>;

    /// Read the active member set of a pool. Used by the
    /// stateless-round-robin dispatcher.
    async fn pool_members(&self, pool_id: u64) -> Result<Vec<PoolMemberInfo>, CoordinatorError>;

    /// Submit `ComputePool.recordDispatch(jobId)` from
    /// `self_address()`.
    async fn record_dispatch(
        &self,
        job_id: u64,
        member: H160,
    ) -> Result<RecordDispatchOutcome, CoordinatorError>;

    /// Submit `ComputePool.completeJob(jobId)` from `self_address()`.
    async fn complete_job(&self, job_id: u64) -> Result<H256, CoordinatorError>;

    /// Submit `ComputePool.failJob(jobId)` from `self_address()`.
    async fn fail_job(&self, job_id: u64) -> Result<H256, CoordinatorError>;
}

/// Convert a block number into the (CM-05 WP-05.1) epoch number.
/// Mirrors `block.number / EPOCH_LENGTH` in `ComputePool.sol`.
pub const EPOCH_LENGTH: u64 = 100;
pub fn epoch_of(block: u64) -> u64 {
    block / EPOCH_LENGTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_of_floor_divides() {
        assert_eq!(epoch_of(0), 0);
        assert_eq!(epoch_of(99), 0);
        assert_eq!(epoch_of(100), 1);
        assert_eq!(epoch_of(199), 1);
        assert_eq!(epoch_of(200), 2);
    }
}
