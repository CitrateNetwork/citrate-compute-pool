//! `Transport` trait + `InProcessTransport` for S0.
//!
//! Workers need to gossip step commitments + coordinate the epoch-
//! boundary aggregation. For S0 we use in-process tokio mpsc
//! broadcasting via shared state. libp2p lands at S1 and slots in
//! behind this same trait.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

use crate::types::{StepCommit, WorkerAddress};

/// Messages exchanged between workers + coordinator on the mesh.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkerMessage {
    /// A worker announces their step commitment. Consumed by the
    /// coordinator for Merkle aggregation; also seen by all other
    /// workers so they can archive for challenge-time look-up.
    StepCommitted(StepCommit),
    /// Coordinator announces that everyone's commits for this
    /// (epoch, step) are in and training can advance.
    StepAdvance { epoch: u32, step: u32 },
    /// Coordinator announces the epoch close; workers stop
    /// producing step commits for this epoch.
    EpochClose { epoch: u32 },
    /// CM-08 pipeline activation forwarding. Stage i sends this to
    /// stage i+1 carrying the computed activation bytes. The
    /// `request_id` binds the activation to a specific in-flight
    /// request so the receiver can match it against the on-chain
    /// request state.
    ///
    /// Activation payload is opaque at this layer — concrete tensor
    /// shapes are decided by the ModelBackend impl. For S0 the
    /// deterministic backend uses a keccak-chained byte vector.
    ///
    /// SECREM-02 6.3 (FUA-COMPUTE-POOL-02): `from_worker` is the
    /// sender's claimed worker address. The libp2p transport drops
    /// any activation whose `from_worker` does not match the
    /// envelope's cryptographically verified signer (same binding
    /// rule as `StepCommitted.worker`), and the pipeline worker
    /// additionally requires `from_worker` to be the on-chain owner
    /// of `from_stage` before accepting the activation.
    PipelineActivation {
        request_id: u64,
        from_stage: u32,
        to_stage: u32,
        from_worker: WorkerAddress,
        payload: Vec<u8>,
    },
}

#[async_trait]
pub trait Transport: Send + Sync {
    /// Broadcast a message to all other peers (including the
    /// coordinator, if the sender isn't the coordinator).
    async fn broadcast(&self, msg: WorkerMessage) -> anyhow::Result<()>;

    /// Pull the next delivered message for this peer. Blocks until
    /// a message arrives. Returns None if the transport is shut
    /// down.
    async fn recv(&self, peer: WorkerAddress) -> Option<WorkerMessage>;

    /// Register a peer so it starts receiving broadcasts.
    async fn register(&self, peer: WorkerAddress);

    /// SECREM-02 6.3 (CITRATE_COMPUTE_POOL-2026-05-31-002): inform
    /// the transport of the worker's current training epoch so it
    /// can bind outgoing envelopes to it and reject inbound
    /// envelopes outside the accepted epoch window. Default no-op
    /// for transports without envelope scoping (in-process tests).
    async fn set_epoch(&self, _epoch: u32) {}
}

/// In-process implementation backed by a single shared queue per
/// peer. Broadcasts fan out to every registered peer except the
/// sender (skip-self convention). Suitable for single-process
/// integration tests that simulate N workers as N tokio tasks.
pub struct InProcessTransport {
    // peer → queue
    state: Arc<Mutex<InProcessState>>,
    notify: Arc<Notify>,
}

#[derive(Default)]
struct InProcessState {
    peers: HashMap<WorkerAddress, VecDeque<WorkerMessage>>,
}

impl InProcessTransport {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Arc::new(Mutex::new(InProcessState::default())),
            notify: Arc::new(Notify::new()),
        })
    }

    /// Get a sender-scoped transport handle — broadcasts from this
    /// handle skip the given peer's own queue. This is the
    /// convention every Transport impl should uphold: don't echo
    /// broadcasts back to the sender.
    pub fn scoped(self: &Arc<Self>, sender: WorkerAddress) -> Arc<ScopedTransport> {
        Arc::new(ScopedTransport {
            inner: Arc::clone(self),
            sender,
        })
    }
}

impl Default for InProcessTransport {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(InProcessState::default())),
            notify: Arc::new(Notify::new()),
        }
    }
}

#[async_trait]
impl Transport for InProcessTransport {
    async fn broadcast(&self, msg: WorkerMessage) -> anyhow::Result<()> {
        let mut state = self.state.lock().await;
        // Default broadcast (no scoping) delivers to every peer.
        for q in state.peers.values_mut() {
            q.push_back(msg.clone());
        }
        drop(state);
        self.notify.notify_waiters();
        Ok(())
    }

    async fn recv(&self, peer: WorkerAddress) -> Option<WorkerMessage> {
        loop {
            {
                let mut state = self.state.lock().await;
                if let Some(q) = state.peers.get_mut(&peer) {
                    if let Some(msg) = q.pop_front() {
                        return Some(msg);
                    }
                }
            }
            // Wait for a notify_waiters before re-checking the queue.
            self.notify.notified().await;
        }
    }

    async fn register(&self, peer: WorkerAddress) {
        let mut state = self.state.lock().await;
        state.peers.entry(peer).or_default();
    }
}

/// Sender-scoped wrapper: broadcasts skip the sender's own queue.
/// Each worker's state machine holds one of these to avoid
/// receiving its own messages.
pub struct ScopedTransport {
    inner: Arc<InProcessTransport>,
    sender: WorkerAddress,
}

#[async_trait]
impl Transport for ScopedTransport {
    async fn broadcast(&self, msg: WorkerMessage) -> anyhow::Result<()> {
        let mut state = self.inner.state.lock().await;
        for (peer, q) in state.peers.iter_mut() {
            if *peer != self.sender {
                q.push_back(msg.clone());
            }
        }
        drop(state);
        self.inner.notify.notify_waiters();
        Ok(())
    }

    async fn recv(&self, peer: WorkerAddress) -> Option<WorkerMessage> {
        self.inner.recv(peer).await
    }

    async fn register(&self, peer: WorkerAddress) {
        self.inner.register(peer).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethereum_types::Address;

    fn addr(b: u8) -> Address {
        Address::repeat_byte(b)
    }

    #[tokio::test]
    async fn broadcast_delivers_to_registered_peers() {
        let transport = InProcessTransport::new();
        let a = addr(1);
        let b = addr(2);
        transport.register(a).await;
        transport.register(b).await;

        let scoped = transport.scoped(a);
        scoped
            .broadcast(WorkerMessage::StepAdvance { epoch: 0, step: 0 })
            .await
            .unwrap();

        // a should NOT receive (sender-scoping skips self).
        // b SHOULD receive.
        let msg = transport.recv(b).await.unwrap();
        matches!(msg, WorkerMessage::StepAdvance { .. });
    }

    #[tokio::test]
    async fn recv_blocks_until_message_arrives() {
        let transport = InProcessTransport::new();
        let a = addr(1);
        let b = addr(2);
        transport.register(a).await;
        transport.register(b).await;

        let t = Arc::clone(&transport);
        let handle = tokio::spawn(async move { t.recv(b).await });

        // Broadcast AFTER the recv task is spawned.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        transport
            .scoped(a)
            .broadcast(WorkerMessage::EpochClose { epoch: 0 })
            .await
            .unwrap();

        let msg = handle.await.unwrap().unwrap();
        matches!(msg, WorkerMessage::EpochClose { .. });
    }
}
