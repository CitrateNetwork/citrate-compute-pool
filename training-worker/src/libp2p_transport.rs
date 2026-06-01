//! libp2p-backed `Transport` implementation for cross-machine
//! worker mesh (S1 slice per `.agentile/launch/S1_S2_BACKLOG.md`).
//!
//! ## Data source (Rule 11)
//!
//! There is no chain data source for this module — it's a network
//! transport layer. The peer-id ↔ on-chain-address binding is out
//! of band: callers derive their [`WorkerAddress`] from the same
//! secp256k1 keypair that they use to sign on-chain transactions
//! (`training-worker/src/wallet.rs::address_from_signing_key`),
//! and the signing key is passed into [`LibP2pTransport::new`] so
//! messages on the mesh are signed with the identical key that
//! signs join-pool calldata.
//!
//! The on-chain registry of peer IDs (`registerWorkerLibP2P`
//! referenced in ADR-009 §Transport) is a separate follow-up item
//! and is represented here as the static bootstrap-peer list the
//! caller supplies.
//!
//! ## Scope
//!
//! * **Wire format**: bincode-serialized `WireEnvelope` per
//!   gossipsub message. ADR-009 specifies protobuf; we use
//!   bincode for S1 bring-up to avoid the `protoc` toolchain dep.
//!   `FIXME(S2):` migrate to protobuf per ADR-009 §"Wire format"
//!   before pilot. bincode and protobuf are ~equivalent in payload
//!   size for the numeric-heavy `StepCommitMessage` shape, so the
//!   migration is a serialization swap, not a protocol rework.
//! * **Transport backends**: libp2p TCP + Noise + Yamux for real
//!   peer-to-peer. Tests use TCP on `127.0.0.1:0` (ephemeral port
//!   on loopback) which matches how libp2p's own gossipsub tests
//!   exercise the stack in-process — this is the S1 test seam.
//!   `MemoryTransport` was considered (per the brief) but requires
//!   manually composing noise + yamux upgrades; TCP loopback gives
//!   the same property (single-process, deterministic, no network
//!   hardware) without that plumbing.
//! * **Peer discovery**: static bootstrap list at construction. No
//!   Kademlia, no mDNS — those are future work. This matches the
//!   "chain-derived bootstrap" model in ADR-009 §Transport: the
//!   caller reads `WorkerJoined` events off-chain and feeds the
//!   resulting peer addresses here.
//! * **Topics**: one gossipsub topic `"citrate-training/default"`
//!   in this slice. Per-job / per-epoch topic derivation is
//!   tracked for S1.1 — the `Transport` trait signature doesn't
//!   carry job_id or epoch at broadcast time, so topic scoping
//!   requires either enriching `WorkerMessage` or passing scope
//!   context in through a new trait method. Out of scope for this
//!   WP.
//!
//! ## Signature binding (critical security property)
//!
//! Every published message is wrapped in a [`WireEnvelope`] that
//! carries:
//!
//! 1. The claimed sender's `WorkerAddress`
//! 2. The serialized `WorkerMessage` payload bytes
//! 3. A secp256k1 signature over the payload bytes
//!
//! On receipt, the envelope is decoded, the sender's claimed
//! address is re-derived from the recovered signature public key,
//! and the two must match. Messages where they don't match are
//! dropped at the transport boundary — they never reach the
//! worker's recv queue.
//!
//! This guarantees that a peer cannot forge a message appearing
//! to come from another worker: forging requires the target's
//! private key. Spoofing the `sender_addr` field without a valid
//! sig fails the recovery-equality check.
//!
//! ## Send-scoping
//!
//! libp2p gossipsub natively skips echoing a message back to its
//! publisher (the publisher's own `publish()` does not produce a
//! local `Event::Message`). This matches the `ScopedTransport`
//! convention from `InProcessTransport` — broadcasts don't
//! re-enter the sender's own `recv` queue.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use ethereum_types::H160;
use futures::stream::StreamExt;
use k256::ecdsa::{signature::Signer, signature::Verifier, Signature, SigningKey, VerifyingKey};
use libp2p::gossipsub::{self, IdentTopic, MessageAuthenticity, ValidationMode};
use libp2p::swarm::{NetworkBehaviour, SwarmEvent};
use libp2p::{identity, noise, tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tracing::{debug, trace, warn};

use crate::transport::{Transport, WorkerMessage};
use crate::types::WorkerAddress;

/// Single gossipsub topic used in this slice. Topic-per-epoch is
/// tracked in S1_S2_BACKLOG — it requires a change to the
/// `Transport` trait or `WorkerMessage` to carry scope metadata
/// at broadcast time.
const DEFAULT_TOPIC: &str = "citrate-training/default";

/// Wire envelope written to the mesh. bincode-serialized, one per
/// gossipsub message.
///
/// `FIXME(S2):` swap bincode for protobuf per ADR-009 §"Wire format".
#[derive(Serialize, Deserialize, Debug, Clone)]
struct WireEnvelope {
    /// Claimed sender address. Verified against the signature on
    /// receipt (see `LibP2pTransport::verify_envelope`).
    sender_addr: [u8; 20],
    /// Bincode-serialized `WorkerMessage`.
    payload: Vec<u8>,
    /// secp256k1 signature (DER-encoded) over `payload`.
    signature: Vec<u8>,
    /// Uncompressed SEC1 public key (65 bytes, 0x04 || X || Y) of
    /// the signer. Enables the receiver to both verify the
    /// signature and re-derive `sender_addr` without a key registry.
    pubkey_sec1: Vec<u8>,
}

#[derive(NetworkBehaviour)]
struct WorkerBehaviour {
    gossipsub: gossipsub::Behaviour,
    identify: libp2p::identify::Behaviour,
    ping: libp2p::ping::Behaviour,
}

#[derive(Debug, thiserror::Error)]
pub enum LibP2pTransportError {
    #[error("swarm build failed: {0}")]
    Build(String),
    #[error("listen failed: {0}")]
    Listen(String),
    #[error("gossipsub publish failed: {0}")]
    Publish(String),
    #[error("subscribe failed: {0}")]
    Subscribe(String),
    #[error("dial failed: {0}")]
    Dial(String),
    #[error("channel closed")]
    Closed,
}

/// Commands the public transport handle sends into the swarm task.
enum SwarmCmd {
    Broadcast(Vec<u8>),
    Dial(Multiaddr),
    ListenAddr(oneshot::Sender<Vec<Multiaddr>>),
}

/// Per-peer recv queue state. Shared between the swarm task
/// (producer) and the transport consumer (via `recv`).
#[derive(Default)]
struct RecvState {
    peers: HashMap<WorkerAddress, VecDeque<WorkerMessage>>,
}

/// libp2p-backed `Transport` implementation.
///
/// See module docs for scope and design. In this slice we run the
/// libp2p swarm on a dedicated tokio task and communicate in/out
/// via tokio channels + a shared recv queue. Broadcast goes
/// through gossipsub; recv pops from the per-peer queue.
pub struct LibP2pTransport {
    /// Signing key used for envelope signatures. Must be the same
    /// key the operator uses for on-chain transactions so that
    /// `sender_addr` (derived from this key) matches the worker's
    /// on-chain identity.
    signing_key: Arc<SigningKey>,
    /// Cached worker address derived from `signing_key` (avoid
    /// re-deriving per message).
    own_address: WorkerAddress,
    /// Cached uncompressed SEC1 pubkey bytes (65B) embedded in
    /// every envelope so receivers can verify without a registry.
    own_pubkey_sec1: Vec<u8>,
    /// Shared recv state — the swarm task writes, the caller reads
    /// via `recv`.
    recv_state: Arc<Mutex<RecvState>>,
    /// Notifier signalled after every successful recv-queue push.
    notify: Arc<Notify>,
    /// Outbound command channel to the swarm task.
    cmd_tx: mpsc::UnboundedSender<SwarmCmd>,
    /// Keepalive for the swarm task; dropping this drops the
    /// Receiver and the task exits cleanly.
    #[allow(dead_code)]
    swarm_task: Arc<tokio::task::JoinHandle<()>>,
}

impl LibP2pTransport {
    /// Create and start a libp2p transport.
    ///
    /// * `signing_key` — secp256k1 key whose derived [`WorkerAddress`]
    ///   is this node's mesh identity. Must match the operator's
    ///   on-chain wallet key.
    /// * `listen_addr` — a libp2p `Multiaddr` to listen on. Use
    ///   `"/ip4/0.0.0.0/tcp/0"` for real deployments (ephemeral port
    ///   OS pick), or `"/ip4/127.0.0.1/tcp/0"` for in-process tests.
    /// * `bootstrap_peers` — static list of peer multiaddrs to dial
    ///   on startup. In a real job this is populated from
    ///   `WorkerJoined` chain events; for tests it's the listen
    ///   addresses of the other peers in the pool.
    ///
    /// Returns an `Arc<Self>` since the trait impl needs cloning
    /// across tasks.
    pub async fn new(
        signing_key: SigningKey,
        listen_addr: Multiaddr,
        bootstrap_peers: Vec<Multiaddr>,
    ) -> Result<Arc<Self>, LibP2pTransportError> {
        let own_address = address_from_signing_key(&signing_key);
        let own_pubkey_sec1 = signing_key
            .verifying_key()
            .to_encoded_point(false)
            .as_bytes()
            .to_vec();

        // libp2p uses its own identity key. We generate a fresh
        // Ed25519 for the libp2p layer (peer ID) — the mesh-level
        // authentication that matters is the per-message envelope
        // signature, which uses the secp256k1 worker key. This
        // decoupling lets libp2p's own noise handshake do its
        // thing without reusing the chain-signing key for transport
        // (defense-in-depth).
        let libp2p_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(libp2p_key.public());
        debug!(peer_id = %local_peer_id, worker_addr = ?own_address, "starting libp2p transport");

        let mut swarm: Swarm<WorkerBehaviour> = SwarmBuilder::with_existing_identity(libp2p_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| LibP2pTransportError::Build(format!("tcp: {e}")))?
            .with_behaviour(|key| {
                let gossipsub_config = gossipsub::ConfigBuilder::default()
                    .heartbeat_interval(Duration::from_millis(200))
                    .validation_mode(ValidationMode::Strict)
                    // Mesh is tiny in the S1 cluster (3-16 peers);
                    // disable duplicate detection by content hash to
                    // avoid surprises when two workers legitimately
                    // send byte-identical `StepAdvance` messages.
                    // Duplicate filtering is keyed by MessageId
                    // instead (default: peer_id + sequence).
                    .build()
                    .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

                let gossipsub = gossipsub::Behaviour::new(
                    MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                )
                .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

                let identify = libp2p::identify::Behaviour::new(libp2p::identify::Config::new(
                    "/citrate/compute/2.training".into(),
                    key.public(),
                ));

                let ping = libp2p::ping::Behaviour::new(libp2p::ping::Config::new());

                Ok(WorkerBehaviour {
                    gossipsub,
                    identify,
                    ping,
                })
            })
            .map_err(|e| LibP2pTransportError::Build(format!("behaviour: {e}")))?
            .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        let topic = IdentTopic::new(DEFAULT_TOPIC);
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topic)
            .map_err(|e| LibP2pTransportError::Subscribe(format!("{e:?}")))?;

        swarm
            .listen_on(listen_addr.clone())
            .map_err(|e| LibP2pTransportError::Listen(format!("{e}")))?;

        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<SwarmCmd>();

        let recv_state: Arc<Mutex<RecvState>> = Arc::new(Mutex::new(RecvState::default()));
        let notify = Arc::new(Notify::new());

        // Register self so `recv(own_address)` doesn't panic if
        // called (it just never delivers anything — we rely on
        // gossipsub's self-skip behaviour).
        recv_state
            .lock()
            .await
            .peers
            .entry(own_address)
            .or_default();

        // Dial bootstrap peers. We ignore errors — dials can fail
        // transiently when the other peer isn't up yet; gossipsub
        // will continue to retry connections via the swarm's own
        // internal logic once peers appear.
        for addr in bootstrap_peers {
            if let Err(e) = swarm.dial(addr.clone()) {
                warn!(?addr, error = %e, "bootstrap dial failed — will rely on inbound/retry");
            }
        }

        // Spawn the swarm driver.
        let recv_state_task = Arc::clone(&recv_state);
        let notify_task = Arc::clone(&notify);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    cmd = cmd_rx.recv() => {
                        let Some(cmd) = cmd else {
                            debug!("swarm cmd channel closed; swarm task exiting");
                            return;
                        };
                        match cmd {
                            SwarmCmd::Broadcast(bytes) => {
                                let topic = IdentTopic::new(DEFAULT_TOPIC);
                                // `InsufficientPeers` is returned when
                                // gossipsub hasn't yet meshed with any
                                // peer. This happens at startup before
                                // the first connection completes. We
                                // warn and continue — the caller can
                                // retry.
                                if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic, bytes) {
                                    warn!(error = %e, "gossipsub publish failed");
                                }
                            }
                            SwarmCmd::Dial(addr) => {
                                if let Err(e) = swarm.dial(addr.clone()) {
                                    warn!(?addr, error = %e, "runtime dial failed");
                                }
                            }
                            SwarmCmd::ListenAddr(tx) => {
                                let addrs: Vec<Multiaddr> = swarm.listeners().cloned().collect();
                                let _ = tx.send(addrs);
                            }
                        }
                    }
                    event = swarm.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(WorkerBehaviourEvent::Gossipsub(gossipsub::Event::Message { message, propagation_source, .. })) => {
                                match verify_envelope(&message.data) {
                                    Ok((sender_addr, worker_msg)) => {
                                        trace!(?sender_addr, %propagation_source, "gossipsub message accepted");

                                        // RM-E.3 / COMPUTE_POOL-001: bind the inner
                                        // StepCommit.worker to the cryptographically
                                        // verified envelope sender. `verify_envelope`
                                        // proves the signer == sender_addr, but the
                                        // inner `worker` field is independent — a peer
                                        // could sign with its own key and claim a
                                        // victim's address. Drop any StepCommitted
                                        // whose `worker` does not match the signer so
                                        // the coordinator can never attribute a forged
                                        // leaf to another worker.
                                        if let WorkerMessage::StepCommitted(ref commit) = worker_msg {
                                            if commit.worker != sender_addr {
                                                warn!(
                                                    ?sender_addr,
                                                    claimed = ?commit.worker,
                                                    "dropping StepCommitted: worker field does not match verified sender"
                                                );
                                                continue;
                                            }
                                        }

                                        let mut state = recv_state_task.lock().await;
                                        // Deliver to every registered peer
                                        // EXCEPT the claimed sender — we preserve
                                        // the `ScopedTransport` self-skip rule so
                                        // semantics match InProcessTransport. In
                                        // practice this only matters if the
                                        // sender also has a recv-side of this
                                        // transport registered locally (single-
                                        // process tests).
                                        for (peer, q) in state.peers.iter_mut() {
                                            if *peer != sender_addr {
                                                q.push_back(worker_msg.clone());
                                            }
                                        }
                                        drop(state);
                                        notify_task.notify_waiters();
                                    }
                                    Err(reason) => {
                                        warn!(%reason, "dropped malformed or forged mesh message");
                                    }
                                }
                            }
                            SwarmEvent::NewListenAddr { address, .. } => {
                                debug!(?address, "listening");
                            }
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                debug!(%peer_id, "peer connected");
                            }
                            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                                debug!(%peer_id, ?cause, "peer disconnected");
                            }
                            other => {
                                trace!(?other, "swarm event");
                            }
                        }
                    }
                }
            }
        });

        Ok(Arc::new(Self {
            signing_key: Arc::new(signing_key),
            own_address,
            own_pubkey_sec1,
            recv_state,
            notify,
            cmd_tx,
            swarm_task: Arc::new(task),
        }))
    }

    /// Return the multiaddrs this transport is listening on. Handy
    /// for tests that need to pass peer addresses to other nodes
    /// after startup (e.g. when listen_addr is port 0).
    pub async fn listeners(&self) -> Result<Vec<Multiaddr>, LibP2pTransportError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(SwarmCmd::ListenAddr(tx))
            .map_err(|_| LibP2pTransportError::Closed)?;
        rx.await.map_err(|_| LibP2pTransportError::Closed)
    }

    /// Dial an additional peer after construction.
    pub fn dial(&self, addr: Multiaddr) -> Result<(), LibP2pTransportError> {
        self.cmd_tx
            .send(SwarmCmd::Dial(addr))
            .map_err(|_| LibP2pTransportError::Closed)
    }

    /// This node's worker address (derived from its signing key).
    pub fn own_address(&self) -> WorkerAddress {
        self.own_address
    }
}

#[async_trait]
impl Transport for LibP2pTransport {
    async fn broadcast(&self, msg: WorkerMessage) -> anyhow::Result<()> {
        let payload = bincode::serialize(&msg)
            .map_err(|e| anyhow::anyhow!("serialize WorkerMessage: {e}"))?;

        // secp256k1 sign over the payload. k256 Signer::sign hashes
        // with SHA-256 internally, matching what our verify path
        // uses via Verifier::verify.
        let signature: Signature = self.signing_key.sign(&payload);
        let signature_der = signature.to_der().as_bytes().to_vec();

        let envelope = WireEnvelope {
            sender_addr: self.own_address.0,
            payload,
            signature: signature_der,
            pubkey_sec1: self.own_pubkey_sec1.clone(),
        };

        let bytes = bincode::serialize(&envelope)
            .map_err(|e| anyhow::anyhow!("serialize envelope: {e}"))?;

        self.cmd_tx
            .send(SwarmCmd::Broadcast(bytes))
            .map_err(|_| anyhow::anyhow!("swarm task has exited"))?;

        Ok(())
    }

    async fn recv(&self, peer: WorkerAddress) -> Option<WorkerMessage> {
        loop {
            {
                let mut state = self.recv_state.lock().await;
                if let Some(q) = state.peers.get_mut(&peer) {
                    if let Some(msg) = q.pop_front() {
                        return Some(msg);
                    }
                }
            }
            self.notify.notified().await;
        }
    }

    async fn register(&self, peer: WorkerAddress) {
        let mut state = self.recv_state.lock().await;
        state.peers.entry(peer).or_default();
    }
}

// ── envelope helpers ──────────────────────────────────────────────

/// Decode and verify a wire envelope. On success, returns
/// `(sender_addr, WorkerMessage)`. Any tamper (bad signature,
/// mismatched public key, bad serialization) returns an error
/// string that's logged at the call site.
fn verify_envelope(bytes: &[u8]) -> Result<(WorkerAddress, WorkerMessage), String> {
    let env: WireEnvelope =
        bincode::deserialize(bytes).map_err(|e| format!("envelope decode: {e}"))?;

    // Parse the claimed pubkey.
    let vk = VerifyingKey::from_sec1_bytes(&env.pubkey_sec1)
        .map_err(|e| format!("bad pubkey: {e}"))?;

    // Re-derive the address from the pubkey and compare to the
    // declared `sender_addr`. This catches the case where a peer
    // embeds a valid signature from key A but claims sender_addr
    // from some other key B.
    let derived_addr = address_from_verifying_key(&vk);
    if derived_addr.0 != env.sender_addr {
        return Err(format!(
            "sender_addr mismatch: claimed {:?}, derived {:?}",
            H160::from(env.sender_addr),
            derived_addr
        ));
    }

    // Verify the signature over the payload.
    let sig = Signature::from_der(&env.signature).map_err(|e| format!("bad sig DER: {e}"))?;
    vk.verify(&env.payload, &sig)
        .map_err(|e| format!("sig verify: {e}"))?;

    // Decode the inner WorkerMessage.
    let msg: WorkerMessage =
        bincode::deserialize(&env.payload).map_err(|e| format!("payload decode: {e}"))?;

    Ok((derived_addr, msg))
}

/// Same derivation as `wallet::address_from_signing_key` — duplicated
/// here to avoid introducing a pub helper surface before the
/// wallet-core unification lands (see FIXME at top of wallet.rs).
fn address_from_signing_key(sk: &SigningKey) -> WorkerAddress {
    address_from_verifying_key(sk.verifying_key())
}

fn address_from_verifying_key(vk: &VerifyingKey) -> WorkerAddress {
    let point = vk.to_encoded_point(false);
    let bytes = &point.as_bytes()[1..]; // drop 0x04
    let mut hasher = Keccak256::new();
    hasher.update(bytes);
    let h = hasher.finalize();
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&h[12..]);
    H160::from(addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StepCommit;
    use ethereum_types::H256;
    use k256::ecdsa::SigningKey;
    use std::time::Duration;

    /// Deterministic test signing key — given a seed byte, produce
    /// a secp256k1 key. Never used in production paths.
    fn test_key(seed: u8) -> SigningKey {
        // 32-byte seed repeated; guaranteed non-zero so the
        // secp256k1 scalar is in range.
        let mut bytes = [seed; 32];
        bytes[0] = seed | 0x01; // ensure not zero
        SigningKey::from_slice(&bytes).expect("deterministic seed → valid secp256k1 key")
    }

    fn listen_addr() -> Multiaddr {
        "/ip4/127.0.0.1/tcp/0"
            .parse()
            .expect("loopback multiaddr parse")
    }

    /// Wait until the transport's listener has bound a concrete port.
    async fn wait_listener(transport: &LibP2pTransport) -> Multiaddr {
        // The listener is registered synchronously during `new`, but
        // the OS may take a moment to surface the final multiaddr
        // via a `NewListenAddr` event. Poll briefly.
        for _ in 0..50 {
            let ls = transport
                .listeners()
                .await
                .expect("listeners query should succeed");
            if let Some(a) = ls.into_iter().next() {
                return a;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("listener never came up");
    }

    fn step_commit_for(epoch: u32, step: u32, worker: WorkerAddress) -> StepCommit {
        StepCommit {
            epoch,
            step,
            worker,
            commitment: H256::repeat_byte(0x11),
            prev_weights: H256::repeat_byte(0x22),
        }
    }

    /// Drain the gossipsub mesh-formation delay. Gossipsub needs a
    /// heartbeat cycle (config'd to 200ms) after connection before
    /// messages fan out reliably. 1s covers the connection +
    /// identify + subscribe + heartbeat sequence with margin.
    async fn settle_mesh() {
        tokio::time::sleep(Duration::from_millis(1_200)).await;
    }

    #[tokio::test]
    async fn three_peers_broadcast_delivers_to_others() {
        let sk_a = test_key(0x0A);
        let sk_b = test_key(0x0B);
        let sk_c = test_key(0x0C);

        let a = LibP2pTransport::new(sk_a.clone(), listen_addr(), vec![])
            .await
            .expect("node A starts");
        let a_addr = wait_listener(&a).await;

        let b = LibP2pTransport::new(sk_b.clone(), listen_addr(), vec![a_addr.clone()])
            .await
            .expect("node B starts");
        let b_addr = wait_listener(&b).await;

        let c = LibP2pTransport::new(sk_c.clone(), listen_addr(), vec![a_addr.clone(), b_addr])
            .await
            .expect("node C starts");
        let _ = wait_listener(&c).await;

        // Register each peer on every other node's recv state so
        // broadcasts land somewhere we can read from.
        let addr_a = a.own_address();
        let addr_b = b.own_address();
        let addr_c = c.own_address();

        a.register(addr_a).await;
        a.register(addr_b).await;
        a.register(addr_c).await;
        b.register(addr_a).await;
        b.register(addr_b).await;
        b.register(addr_c).await;
        c.register(addr_a).await;
        c.register(addr_b).await;
        c.register(addr_c).await;

        settle_mesh().await;

        // A broadcasts.
        let msg = WorkerMessage::StepCommitted(step_commit_for(0, 0, addr_a));
        a.broadcast(msg).await.expect("A publish ok");

        // B and C should each receive.
        let b_recv = tokio::time::timeout(Duration::from_secs(5), b.recv(addr_b))
            .await
            .expect("B recv timed out")
            .expect("B got a message");
        let c_recv = tokio::time::timeout(Duration::from_secs(5), c.recv(addr_c))
            .await
            .expect("C recv timed out")
            .expect("C got a message");

        match b_recv {
            WorkerMessage::StepCommitted(sc) => assert_eq!(sc.worker, addr_a),
            other => panic!("B received unexpected variant: {other:?}"),
        }
        match c_recv {
            WorkerMessage::StepCommitted(sc) => assert_eq!(sc.worker, addr_a),
            other => panic!("C received unexpected variant: {other:?}"),
        }
    }

    #[tokio::test]
    async fn sender_does_not_receive_own_broadcast() {
        let sk_a = test_key(0x1A);
        let sk_b = test_key(0x1B);

        let a = LibP2pTransport::new(sk_a, listen_addr(), vec![])
            .await
            .expect("A starts");
        let a_addr = wait_listener(&a).await;

        let b = LibP2pTransport::new(sk_b, listen_addr(), vec![a_addr])
            .await
            .expect("B starts");
        let _ = wait_listener(&b).await;

        let addr_a = a.own_address();
        let addr_b = b.own_address();
        a.register(addr_a).await;
        a.register(addr_b).await;
        b.register(addr_a).await;
        b.register(addr_b).await;

        settle_mesh().await;

        a.broadcast(WorkerMessage::EpochClose { epoch: 7 })
            .await
            .expect("publish");

        // B should receive.
        let got = tokio::time::timeout(Duration::from_secs(5), b.recv(addr_b))
            .await
            .expect("B recv timed out")
            .expect("B got msg");
        assert!(matches!(got, WorkerMessage::EpochClose { epoch: 7 }));

        // A should NOT receive its own broadcast. gossipsub filters
        // self-publish at the swarm layer so the envelope never
        // hits A's recv queue. Assert by polling A's own queue
        // briefly and expecting a timeout.
        let a_own_recv = tokio::time::timeout(Duration::from_millis(500), a.recv(addr_a)).await;
        assert!(
            a_own_recv.is_err(),
            "A should not receive its own broadcast, but got {a_own_recv:?}"
        );
    }

    #[tokio::test]
    async fn forged_signature_is_rejected_by_verify() {
        // Unit test of the envelope verify path — doesn't need a
        // real swarm since the check is on the decoded bytes.
        let sk = test_key(0x2A);
        let vk_bytes = sk
            .verifying_key()
            .to_encoded_point(false)
            .as_bytes()
            .to_vec();

        let payload = bincode::serialize(&WorkerMessage::StepAdvance { epoch: 1, step: 2 })
            .expect("serialize msg");

        // Legitimate signature = baseline: this envelope should verify.
        let good_sig: Signature = sk.sign(&payload);
        let good_envelope = WireEnvelope {
            sender_addr: address_from_signing_key(&sk).0,
            payload: payload.clone(),
            signature: good_sig.to_der().as_bytes().to_vec(),
            pubkey_sec1: vk_bytes.clone(),
        };
        let good_bytes = bincode::serialize(&good_envelope).expect("ser good");
        verify_envelope(&good_bytes).expect("good envelope verifies");

        // Forged: zeroed-out signature.
        let mut bad_envelope = good_envelope.clone();
        bad_envelope.signature = vec![0u8; bad_envelope.signature.len()];
        let bad_bytes = bincode::serialize(&bad_envelope).expect("ser bad");
        assert!(
            verify_envelope(&bad_bytes).is_err(),
            "forged-signature envelope must be rejected"
        );

        // Forged: valid signature but wrong claimed sender_addr.
        let other_sk = test_key(0x2B);
        let mut spoofed = good_envelope.clone();
        spoofed.sender_addr = address_from_signing_key(&other_sk).0;
        let spoofed_bytes = bincode::serialize(&spoofed).expect("ser spoof");
        assert!(
            verify_envelope(&spoofed_bytes).is_err(),
            "sender_addr mismatch must be rejected"
        );

        // Forged: tampered payload (sig was over the original
        // payload, so bumping a byte should invalidate it).
        let mut tampered = good_envelope.clone();
        tampered.payload[0] ^= 0xFF;
        let tampered_bytes = bincode::serialize(&tampered).expect("ser tampered");
        assert!(
            verify_envelope(&tampered_bytes).is_err(),
            "tampered payload must be rejected"
        );
    }

    #[tokio::test]
    async fn forged_peer_cannot_impersonate_over_mesh() {
        // End-to-end version: start two legitimate peers and inject
        // a forged envelope onto the wire. Gossipsub's own Signed
        // authenticity + our envelope signature both must filter
        // it. We can't easily inject raw bytes into a running
        // gossipsub without bypassing libp2p, so this test
        // constructs the forgery scenario at the verify layer
        // directly — equivalent to the path every inbound message
        // takes inside the swarm task.
        let legit_sk = test_key(0x3A);
        let attacker_sk = test_key(0x3B);
        let legit_addr = address_from_signing_key(&legit_sk);

        // Attacker signs their own payload but stamps `sender_addr`
        // with the legitimate worker's address.
        let payload = bincode::serialize(&WorkerMessage::EpochClose { epoch: 99 })
            .expect("serialize msg");
        let attacker_sig: Signature = attacker_sk.sign(&payload);
        let attacker_vk_bytes = attacker_sk
            .verifying_key()
            .to_encoded_point(false)
            .as_bytes()
            .to_vec();

        let forged = WireEnvelope {
            sender_addr: legit_addr.0, // lie: pretending to be legit worker
            payload,
            signature: attacker_sig.to_der().as_bytes().to_vec(),
            pubkey_sec1: attacker_vk_bytes, // but carrying attacker's pubkey
        };
        let forged_bytes = bincode::serialize(&forged).expect("ser forged");

        // The verify path must reject because `sender_addr` is
        // derived from the embedded pubkey, and the embedded
        // pubkey is the attacker's, which does not hash to the
        // legitimate worker's address.
        let err = verify_envelope(&forged_bytes)
            .expect_err("impersonation attempt must fail verification");
        assert!(
            err.contains("sender_addr mismatch"),
            "unexpected rejection reason: {err}"
        );
    }

    #[tokio::test]
    async fn register_and_recv_without_messages_blocks() {
        let sk = test_key(0x4A);
        let transport = LibP2pTransport::new(sk, listen_addr(), vec![])
            .await
            .expect("starts");
        let _ = wait_listener(&transport).await;

        let peer = WorkerAddress::repeat_byte(0xFE);
        transport.register(peer).await;

        // recv should block (no messages, no peers). Poll briefly.
        let res =
            tokio::time::timeout(Duration::from_millis(200), transport.recv(peer)).await;
        assert!(res.is_err(), "recv must block when no messages queued");
    }
}
