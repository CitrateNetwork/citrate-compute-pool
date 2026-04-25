//! Coordinator error surface.

use thiserror::Error;

/// All ways the daemon can fail to handle one event.
#[derive(Debug, Error)]
pub enum CoordinatorError {
    /// `coordinator_for(pool, epoch)` returned someone other than us;
    /// the daemon silently moves on.
    #[error("not the coordinator for this epoch")]
    NotCoordinator,

    /// Chain RPC call failed (read or write).
    #[error("chain: {0}")]
    Chain(String),

    /// Provider HTTPS dispatch failed (timeout, 5xx, malformed body).
    #[error("provider failed: {0}")]
    ProviderFailed(String),

    /// Pool member set is empty — nothing to dispatch to.
    #[error("pool {0} has no active members")]
    NoMembers(u64),

    /// We tried to dispatch to a member whose endpoint isn't in our
    /// config map. Operator deployment issue.
    #[error("no endpoint configured for member {0:?}")]
    UnknownMemberEndpoint(ethereum_types::H160),

    /// Anything we genuinely don't expect.
    #[error("internal: {0}")]
    Internal(String),
}
