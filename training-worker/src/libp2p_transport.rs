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
//! * **Topics**: one gossipsub topic PER JOB, derived as
//!   `citrate-training/job-<job_id>` from the [`MeshScope`] given
//!   at construction (SECREM-02 6.3 closes the single-shared-topic
//!   deferral from the 2026-05-31 audit). The topic string is also
//!   bound into every envelope's signed [`EnvelopeContext`].
//!
//! ## Signature binding (critical security property)
//!
//! SECREM-02 6.3 / CITRATE_COMPUTE_POOL-2026-05-31-002: every
//! published message is wrapped in a versioned [`WireEnvelope`]
//! that carries:
//!
//! 1. A signed [`EnvelopeContext`] — wire version, job id, epoch,
//!    per-sender nonce, topic string, and send timestamp
//! 2. The claimed sender's `WorkerAddress`
//! 3. The serialized `WorkerMessage` payload bytes
//! 4. A secp256k1 signature over a domain-separated digest of the
//!    context AND the payload bytes
//!
//! On receipt, verification fails closed unless ALL hold:
//!
//! * the wire version is the supported [`ENVELOPE_WIRE_VERSION`]
//! * the context's `topic` and `job_id` match the receiving
//!   transport's own scope (cross-topic / cross-job replays die
//!   here — the signature covers the context, so an attacker
//!   cannot rewrite the context to fit another mesh)
//! * the context's `epoch` is within ±[`EPOCH_TOLERANCE`] of the
//!   receiver's current epoch (cross-epoch replays die here)
//! * the context's `sent_at_ms` is within the freshness window
//! * the signer's re-derived address equals `sender_addr`
//! * the signature verifies over (domain tag ‖ context ‖ payload)
//! * the `(sender, nonce)` pair has not been seen inside the
//!   bounded replay window (byte-identical replays die here)
//!
//! Inner-message sender binding (RM-E.3 + FUA-COMPUTE-POOL-02):
//! `StepCommitted.worker` and `PipelineActivation.from_worker`
//! must equal the verified envelope signer, or the message is
//! dropped at the transport boundary.
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

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

/// Wire-format version of [`WireEnvelope`]. v1 (pre-SECREM-02 6.3)
/// signed only the bare payload with no context binding and is NOT
/// accepted — there is no deployed v1 network to interoperate with,
/// so verification fails closed on any version other than this one.
const ENVELOPE_WIRE_VERSION: u8 = 2;

/// Domain separator mixed into every envelope signature so a mesh
/// envelope signature can never be confused with (or replayed as)
/// any other secp256k1 signature this key produces (e.g. calldata).
const ENVELOPE_DOMAIN_TAG: &[u8] = b"citrate-mesh-envelope-v2";

/// Receiver-side epoch window: an envelope is accepted if its
/// context epoch is within ± this of the receiver's current epoch
/// (peers race slightly around epoch boundaries).
const EPOCH_TOLERANCE: u32 = 1;

/// Maximum accepted envelope age. Replays older than this are
/// rejected even after their nonce has been evicted from the
/// bounded [`ReplayGuard`] window.
const MAX_ENVELOPE_AGE_MS: u64 = 5 * 60 * 1000;

/// Maximum accepted forward clock skew for `sent_at_ms`.
const MAX_ENVELOPE_FUTURE_SKEW_MS: u64 = 30 * 1000;

/// Capacity of the per-transport `(sender, nonce)` replay window.
const REPLAY_WINDOW: usize = 4096;

/// Mesh scope this transport participates in. Every envelope is
/// bound to this scope; envelopes from other scopes are rejected.
#[derive(Clone, Copy, Debug)]
pub struct MeshScope {
    /// On-chain training-job id (`ComputePoolTraining` job handle).
    pub job_id: u64,
}

/// Derive the per-job gossipsub topic name.
fn topic_for_job(job_id: u64) -> String {
    format!("citrate-training/job-{job_id}")
}

/// Signed envelope context (SECREM-02 6.3). Serialized inside
/// [`WireEnvelope`] AND covered by the envelope signature, so none
/// of these fields can be rewritten without invalidating the sig.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
struct EnvelopeContext {
    /// Wire-format version. Must equal [`ENVELOPE_WIRE_VERSION`].
    version: u8,
    /// Job this envelope belongs to. Must match the receiver's scope.
    job_id: u64,
    /// Sender's training epoch at send time. Must be within
    /// [`EPOCH_TOLERANCE`] of the receiver's current epoch.
    epoch: u32,
    /// Per-sender monotonically increasing nonce. A `(sender,
    /// nonce)` pair is accepted at most once per replay window.
    nonce: u64,
    /// Topic the envelope was published to. Must match the topic
    /// the receiver is subscribed to (cross-topic replay guard).
    topic: String,
    /// Sender wall-clock at send time (ms since Unix epoch).
    sent_at_ms: u64,
}

/// Wire envelope written to the mesh. bincode-serialized, one per
/// gossipsub message. Version 2 (see [`ENVELOPE_WIRE_VERSION`]);
/// the leading `context.version` byte is the first wire byte.
///
/// `FIXME(S2):` swap bincode for protobuf per ADR-009 §"Wire format".
#[derive(Serialize, Deserialize, Debug, Clone)]
struct WireEnvelope {
    /// Signed scope binding (job/epoch/nonce/topic/timestamp).
    context: EnvelopeContext,
    /// Claimed sender address. Verified against the signature on
    /// receipt (see `verify_envelope`).
    sender_addr: [u8; 20],
    /// Bincode-serialized `WorkerMessage`.
    payload: Vec<u8>,
    /// secp256k1 signature (DER-encoded) over
    /// `envelope_signing_bytes(context, payload)`.
    signature: Vec<u8>,
    /// Uncompressed SEC1 public key (65 bytes, 0x04 || X || Y) of
    /// the signer. Enables the receiver to both verify the
    /// signature and re-derive `sender_addr` without a key registry.
    pubkey_sec1: Vec<u8>,
}

/// The exact byte string the envelope signature covers: domain tag
/// ‖ bincode(context) ‖ payload. Binding the context into the sig
/// is what makes cross-job/epoch/topic replay impossible — an
/// attacker cannot rewrite the context to fit another mesh without
/// the sender's key.
fn envelope_signing_bytes(ctx: &EnvelopeContext, payload: &[u8]) -> Vec<u8> {
    let ctx_bytes = bincode::serialize(ctx).expect("EnvelopeContext serialize cannot fail");
    let mut buf = Vec::with_capacity(ENVELOPE_DOMAIN_TAG.len() + ctx_bytes.len() + payload.len());
    buf.extend_from_slice(ENVELOPE_DOMAIN_TAG);
    buf.extend_from_slice(&ctx_bytes);
    buf.extend_from_slice(payload);
    buf
}

/// Current wall clock in ms since the Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Bounded `(sender, nonce)` replay window (FIFO eviction). Lives
/// in the swarm task — one per transport, single consumer.
struct ReplayGuard {
    seen: HashSet<(WorkerAddress, u64)>,
    order: VecDeque<(WorkerAddress, u64)>,
    cap: usize,
}

impl ReplayGuard {
    fn new(cap: usize) -> Self {
        Self {
            seen: HashSet::new(),
            order: VecDeque::new(),
            cap,
        }
    }

    /// Record `(sender, nonce)`. Returns `false` if it was already
    /// inside the window (replay), `true` if it's fresh.
    fn observe(&mut self, sender: WorkerAddress, nonce: u64) -> bool {
        let key = (sender, nonce);
        if self.seen.contains(&key) {
            return false;
        }
        if self.order.len() >= self.cap {
            if let Some(evicted) = self.order.pop_front() {
                self.seen.remove(&evicted);
            }
        }
        self.seen.insert(key);
        self.order.push_back(key);
        true
    }
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
    /// Mesh scope (job id) this transport is bound to.
    scope: MeshScope,
    /// Per-job topic string derived from `scope`.
    topic: String,
    /// Current training epoch — written by `set_epoch`, read by
    /// `broadcast` (outbound context) and the swarm task (inbound
    /// epoch-window check).
    current_epoch: Arc<AtomicU32>,
    /// Outbound envelope nonce. Seeded from the wall clock so two
    /// process restarts inside the replay window don't collide.
    nonce: AtomicU64,
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
    /// * `scope` — the job this mesh belongs to. Drives the per-job
    ///   gossipsub topic and the signed envelope context; envelopes
    ///   from any other job/topic are rejected (SECREM-02 6.3).
    ///
    /// Returns an `Arc<Self>` since the trait impl needs cloning
    /// across tasks.
    pub async fn new(
        signing_key: SigningKey,
        listen_addr: Multiaddr,
        bootstrap_peers: Vec<Multiaddr>,
        scope: MeshScope,
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

        let topic_name = topic_for_job(scope.job_id);
        let topic = IdentTopic::new(topic_name.clone());
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
        let current_epoch = Arc::new(AtomicU32::new(0));
        let epoch_task = Arc::clone(&current_epoch);
        let topic_task = topic_name.clone();
        let scope_job_id = scope.job_id;
        let task = tokio::spawn(async move {
            // Bounded (sender, nonce) replay window — swarm task is
            // the single consumer of inbound envelopes.
            let mut replay_guard = ReplayGuard::new(REPLAY_WINDOW);
            let own_topic_hash = IdentTopic::new(topic_task.clone()).hash();
            loop {
                tokio::select! {
                    cmd = cmd_rx.recv() => {
                        let Some(cmd) = cmd else {
                            debug!("swarm cmd channel closed; swarm task exiting");
                            return;
                        };
                        match cmd {
                            SwarmCmd::Broadcast(bytes) => {
                                let topic = IdentTopic::new(topic_task.clone());
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
                                // Belt-and-braces: we only subscribe to our own
                                // per-job topic, but assert the delivery topic
                                // anyway so a future multi-topic subscription
                                // can't silently widen the accept surface.
                                if message.topic != own_topic_hash {
                                    warn!(topic = %message.topic, "dropping message from foreign gossipsub topic");
                                    continue;
                                }
                                match verify_envelope(
                                    &message.data,
                                    &topic_task,
                                    scope_job_id,
                                    epoch_task.load(Ordering::Acquire),
                                    now_ms(),
                                    &mut replay_guard,
                                ) {
                                    Ok((sender_addr, worker_msg)) => {
                                        trace!(?sender_addr, %propagation_source, "gossipsub message accepted");

                                        // RM-E.3 / COMPUTE_POOL-001 + SECREM-02 6.3 /
                                        // FUA-COMPUTE-POOL-02: bind inner sender-
                                        // identity fields (`StepCommitted.worker`,
                                        // `PipelineActivation.from_worker`) to the
                                        // cryptographically verified envelope signer.
                                        if let Err(reason) = enforce_sender_binding(sender_addr, &worker_msg) {
                                            warn!(?sender_addr, %reason, "dropping mesh message: sender binding violated");
                                            continue;
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
            scope,
            topic: topic_name,
            current_epoch,
            nonce: AtomicU64::new(now_ms()),
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

        // SECREM-02 6.3: bind the envelope to this transport's
        // job/epoch/topic plus a fresh nonce + timestamp, and sign
        // context AND payload together (domain-separated). See
        // `envelope_signing_bytes` for the exact byte layout.
        let context = EnvelopeContext {
            version: ENVELOPE_WIRE_VERSION,
            job_id: self.scope.job_id,
            epoch: self.current_epoch.load(Ordering::Acquire),
            nonce: self.nonce.fetch_add(1, Ordering::Relaxed),
            topic: self.topic.clone(),
            sent_at_ms: now_ms(),
        };

        // secp256k1 sign over (domain tag ‖ context ‖ payload).
        // k256 Signer::sign hashes with SHA-256 internally, matching
        // what our verify path uses via Verifier::verify.
        let signature: Signature = self
            .signing_key
            .sign(&envelope_signing_bytes(&context, &payload));
        let signature_der = signature.to_der().as_bytes().to_vec();

        let envelope = WireEnvelope {
            context,
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

    async fn set_epoch(&self, epoch: u32) {
        self.current_epoch.store(epoch, Ordering::Release);
    }
}

// ── envelope helpers ──────────────────────────────────────────────

/// Decode and verify a wire envelope. On success, returns
/// `(sender_addr, WorkerMessage)`. Any tamper (bad signature,
/// mismatched public key, bad serialization) returns an error
/// string that's logged at the call site.
/// FUA-COMPUTE-POOL-01: hard limit on a decoded mesh message. bincode honors
/// this limit while reading length prefixes, so a hostile envelope cannot make
/// the decoder pre-allocate gigabytes before any signature/auth check runs.
const MAX_ENVELOPE_BYTES: u64 = 4 * 1024 * 1024;

/// bincode options matching the free-function wire format (fixint, little-endian,
/// reject-trailing) but with a byte limit added (FUA-COMPUTE-POOL-01).
fn bincode_limited() -> impl bincode::Options {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_ENVELOPE_BYTES)
}

/// Decode and verify a wire envelope against the receiving
/// transport's scope. Fail-closed: any one failing check rejects
/// the envelope (SECREM-02 6.3, CITRATE_COMPUTE_POOL-2026-05-31-002).
///
/// * `expected_topic` / `expected_job` — the receiver's own scope.
/// * `current_epoch` — the receiver's training epoch (window check).
/// * `now` — receiver wall clock in ms (freshness window).
/// * `replay_guard` — bounded `(sender, nonce)` window; a nonce is
///   only recorded AFTER the signature verifies, so unauthenticated
///   garbage cannot poison or evict window entries.
fn verify_envelope(
    bytes: &[u8],
    expected_topic: &str,
    expected_job: u64,
    current_epoch: u32,
    now: u64,
    replay_guard: &mut ReplayGuard,
) -> Result<(WorkerAddress, WorkerMessage), String> {
    use bincode::Options;
    // Reject an oversized frame up front, then decode under a byte limit (bincode
    // checks length prefixes against the remaining limit → no pre-alloc OOM).
    if bytes.len() as u64 > MAX_ENVELOPE_BYTES {
        return Err(format!(
            "envelope too large: {} > {MAX_ENVELOPE_BYTES} cap",
            bytes.len()
        ));
    }
    let env: WireEnvelope = bincode_limited()
        .deserialize(bytes)
        .map_err(|e| format!("envelope decode: {e}"))?;

    // ── Context binding (all fields are under the signature) ──
    let ctx = &env.context;
    if ctx.version != ENVELOPE_WIRE_VERSION {
        return Err(format!(
            "unsupported envelope version {} (expected {ENVELOPE_WIRE_VERSION})",
            ctx.version
        ));
    }
    if ctx.topic != expected_topic {
        return Err(format!(
            "topic mismatch: envelope bound to {:?}, receiving on {expected_topic:?}",
            ctx.topic
        ));
    }
    if ctx.job_id != expected_job {
        return Err(format!(
            "job mismatch: envelope bound to job {}, receiver scoped to job {expected_job}",
            ctx.job_id
        ));
    }
    let epoch_lo = current_epoch.saturating_sub(EPOCH_TOLERANCE);
    let epoch_hi = current_epoch.saturating_add(EPOCH_TOLERANCE);
    if ctx.epoch < epoch_lo || ctx.epoch > epoch_hi {
        return Err(format!(
            "epoch {} outside accepted window [{epoch_lo}, {epoch_hi}]",
            ctx.epoch
        ));
    }
    if now.saturating_sub(ctx.sent_at_ms) > MAX_ENVELOPE_AGE_MS {
        return Err(format!(
            "stale envelope: sent_at_ms={} is older than {MAX_ENVELOPE_AGE_MS}ms",
            ctx.sent_at_ms
        ));
    }
    if ctx.sent_at_ms.saturating_sub(now) > MAX_ENVELOPE_FUTURE_SKEW_MS {
        return Err(format!(
            "future-dated envelope: sent_at_ms={} exceeds {MAX_ENVELOPE_FUTURE_SKEW_MS}ms skew",
            ctx.sent_at_ms
        ));
    }

    // ── Sender authentication ──
    // Parse the claimed pubkey.
    let vk =
        VerifyingKey::from_sec1_bytes(&env.pubkey_sec1).map_err(|e| format!("bad pubkey: {e}"))?;

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

    // Verify the signature over (domain tag ‖ context ‖ payload) —
    // the context is INSIDE the signed bytes, so none of the checks
    // above can be satisfied by rewriting context fields.
    let sig = Signature::from_der(&env.signature).map_err(|e| format!("bad sig DER: {e}"))?;
    vk.verify(&envelope_signing_bytes(ctx, &env.payload), &sig)
        .map_err(|e| format!("sig verify: {e}"))?;

    // ── Replay window (post-auth so only signed nonces count) ──
    if !replay_guard.observe(derived_addr, ctx.nonce) {
        return Err(format!(
            "replayed envelope: nonce {} already seen from {derived_addr:?}",
            ctx.nonce
        ));
    }

    // Decode the inner WorkerMessage under the same byte limit (the payload was
    // bounded by the envelope cap above, but decode it limited too for safety).
    let msg: WorkerMessage = bincode_limited()
        .deserialize(&env.payload)
        .map_err(|e| format!("payload decode: {e}"))?;

    Ok((derived_addr, msg))
}

/// Bind inner-message sender-identity fields to the verified
/// envelope signer. A peer signs with its OWN key, so any identity
/// it claims inside the payload must match that key's address:
///
/// * `StepCommitted.worker` (RM-E.3 / COMPUTE_POOL-001)
/// * `PipelineActivation.from_worker` (SECREM-02 6.3 / FUA-COMPUTE-POOL-02)
fn enforce_sender_binding(sender: WorkerAddress, msg: &WorkerMessage) -> Result<(), String> {
    match msg {
        WorkerMessage::StepCommitted(commit) if commit.worker != sender => Err(format!(
            "StepCommitted.worker {:?} does not match verified sender {sender:?}",
            commit.worker
        )),
        WorkerMessage::PipelineActivation { from_worker, .. } if *from_worker != sender => {
            Err(format!(
                "PipelineActivation.from_worker {from_worker:?} does not match verified sender {sender:?}"
            ))
        }

        _ => Ok(()),
    }
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

    /// Shared test scope — all peers of one mesh must use the same
    /// job id (it drives the topic + envelope binding).
    const TEST_JOB: u64 = 7;

    fn test_scope() -> MeshScope {
        MeshScope { job_id: TEST_JOB }
    }

    fn test_topic() -> String {
        topic_for_job(TEST_JOB)
    }

    /// A fresh, valid v2 context for `test_scope()` at epoch 0.
    fn test_ctx(nonce: u64) -> EnvelopeContext {
        EnvelopeContext {
            version: ENVELOPE_WIRE_VERSION,
            job_id: TEST_JOB,
            epoch: 0,
            nonce,
            topic: test_topic(),
            sent_at_ms: now_ms(),
        }
    }

    /// Build a correctly signed v2 envelope for `msg` under `ctx`.
    /// This is the re-signed fixture helper — the v1 fixtures (bare
    /// payload signature, no context) are intentionally no longer
    /// constructible against the public surface.
    fn signed_envelope(sk: &SigningKey, msg: &WorkerMessage, ctx: EnvelopeContext) -> WireEnvelope {
        let payload = bincode::serialize(msg).expect("serialize msg");
        let sig: Signature = sk.sign(&envelope_signing_bytes(&ctx, &payload));
        WireEnvelope {
            context: ctx,
            sender_addr: address_from_signing_key(sk).0,
            payload,
            signature: sig.to_der().as_bytes().to_vec(),
            pubkey_sec1: sk
                .verifying_key()
                .to_encoded_point(false)
                .as_bytes()
                .to_vec(),
        }
    }

    /// Run the verify path with this transport-equivalent receiver
    /// scope (topic/job of `test_scope()`, epoch 0, now).
    fn verify_with(
        bytes: &[u8],
        guard: &mut ReplayGuard,
    ) -> Result<(WorkerAddress, WorkerMessage), String> {
        verify_envelope(bytes, &test_topic(), TEST_JOB, 0, now_ms(), guard)
    }

    fn fresh_guard() -> ReplayGuard {
        ReplayGuard::new(REPLAY_WINDOW)
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

        let a = LibP2pTransport::new(sk_a.clone(), listen_addr(), vec![], test_scope())
            .await
            .expect("node A starts");
        let a_addr = wait_listener(&a).await;

        let b = LibP2pTransport::new(
            sk_b.clone(),
            listen_addr(),
            vec![a_addr.clone()],
            test_scope(),
        )
        .await
        .expect("node B starts");
        let b_addr = wait_listener(&b).await;

        let c = LibP2pTransport::new(
            sk_c.clone(),
            listen_addr(),
            vec![a_addr.clone(), b_addr],
            test_scope(),
        )
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

        let a = LibP2pTransport::new(sk_a, listen_addr(), vec![], test_scope())
            .await
            .expect("A starts");
        let a_addr = wait_listener(&a).await;

        let b = LibP2pTransport::new(sk_b, listen_addr(), vec![a_addr], test_scope())
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
        // Fixtures re-signed for wire v2 (context inside the sig).
        let sk = test_key(0x2A);
        let msg = WorkerMessage::StepAdvance { epoch: 1, step: 2 };

        // Legitimate signature = baseline: this envelope should verify.
        let good_envelope = signed_envelope(&sk, &msg, test_ctx(1));
        let good_bytes = bincode::serialize(&good_envelope).expect("ser good");
        verify_with(&good_bytes, &mut fresh_guard()).expect("good envelope verifies");

        // FUA-COMPUTE-POOL-01: an over-cap frame is refused before any decode/alloc.
        let oversized = vec![0u8; (MAX_ENVELOPE_BYTES as usize) + 1];
        let too_big = verify_with(&oversized, &mut fresh_guard())
            .expect_err("oversized frame must be refused");
        assert!(too_big.contains("too large"), "err was: {too_big}");

        // Forged: zeroed-out signature.
        let mut bad_envelope = good_envelope.clone();
        bad_envelope.signature = vec![0u8; bad_envelope.signature.len()];
        let bad_bytes = bincode::serialize(&bad_envelope).expect("ser bad");
        assert!(
            verify_with(&bad_bytes, &mut fresh_guard()).is_err(),
            "forged-signature envelope must be rejected"
        );

        // Forged: valid signature but wrong claimed sender_addr.
        let other_sk = test_key(0x2B);
        let mut spoofed = good_envelope.clone();
        spoofed.sender_addr = address_from_signing_key(&other_sk).0;
        let spoofed_bytes = bincode::serialize(&spoofed).expect("ser spoof");
        assert!(
            verify_with(&spoofed_bytes, &mut fresh_guard()).is_err(),
            "sender_addr mismatch must be rejected"
        );

        // Forged: tampered payload (sig was over the original
        // payload, so bumping a byte should invalidate it).
        let mut tampered = good_envelope.clone();
        tampered.payload[0] ^= 0xFF;
        let tampered_bytes = bincode::serialize(&tampered).expect("ser tampered");
        assert!(
            verify_with(&tampered_bytes, &mut fresh_guard()).is_err(),
            "tampered payload must be rejected"
        );
    }

    /// SECREM-02 6.3 tripwires (CITRATE_COMPUTE_POOL-2026-05-31-002):
    /// every context-binding check must individually reject.
    #[tokio::test]
    async fn envelope_context_binding_rejects_cross_scope_replay() {
        let sk = test_key(0x6A);
        let msg = WorkerMessage::StepAdvance { epoch: 0, step: 1 };

        // (a) Cross-TOPIC replay: envelope signed for another topic
        // is rejected on ours — even though the signature itself is
        // valid for that other topic.
        let mut ctx = test_ctx(10);
        ctx.topic = topic_for_job(999);
        let env = signed_envelope(&sk, &msg, ctx);
        let bytes = bincode::serialize(&env).expect("ser");
        let err = verify_with(&bytes, &mut fresh_guard()).expect_err("cross-topic must fail");
        assert!(err.contains("topic mismatch"), "err was: {err}");

        // (b) Cross-JOB replay: context job differs from receiver's.
        let mut ctx = test_ctx(11);
        ctx.job_id = TEST_JOB + 1;
        let env = signed_envelope(&sk, &msg, ctx);
        let bytes = bincode::serialize(&env).expect("ser");
        let err = verify_with(&bytes, &mut fresh_guard()).expect_err("cross-job must fail");
        assert!(err.contains("job mismatch"), "err was: {err}");

        // (c) Cross-EPOCH replay: epoch outside the ±1 window of the
        // receiver's current epoch (receiver at 0 here).
        let mut ctx = test_ctx(12);
        ctx.epoch = 5;
        let env = signed_envelope(&sk, &msg, ctx);
        let bytes = bincode::serialize(&env).expect("ser");
        let err = verify_with(&bytes, &mut fresh_guard()).expect_err("cross-epoch must fail");
        assert!(err.contains("epoch"), "err was: {err}");

        // (d) STALE replay: envelope older than the freshness window.
        let mut ctx = test_ctx(13);
        ctx.sent_at_ms = now_ms() - MAX_ENVELOPE_AGE_MS - 1_000;
        let env = signed_envelope(&sk, &msg, ctx);
        let bytes = bincode::serialize(&env).expect("ser");
        let err = verify_with(&bytes, &mut fresh_guard()).expect_err("stale must fail");
        assert!(err.contains("stale"), "err was: {err}");

        // (e) Context REWRITE: take a valid cross-job envelope and
        // rewrite its context to fit our scope — the signature was
        // over the original context, so verification must fail.
        // This is exactly the cross-job replay an attacker would
        // attempt against the v1 (payload-only-signature) format.
        let mut foreign_ctx = test_ctx(14);
        foreign_ctx.job_id = TEST_JOB + 1;
        foreign_ctx.topic = topic_for_job(TEST_JOB + 1);
        let mut env = signed_envelope(&sk, &msg, foreign_ctx);
        env.context.job_id = TEST_JOB; // attacker rewrite
        env.context.topic = test_topic();
        let bytes = bincode::serialize(&env).expect("ser");
        let err = verify_with(&bytes, &mut fresh_guard())
            .expect_err("rewritten context must fail signature verification");
        assert!(err.contains("sig verify"), "err was: {err}");
    }

    /// SECREM-02 6.3: the legacy v1 wire shape (no context, bare
    /// payload signature) must be rejected.
    #[tokio::test]
    async fn legacy_v1_envelope_is_rejected() {
        #[derive(Serialize)]
        struct WireEnvelopeV1 {
            sender_addr: [u8; 20],
            payload: Vec<u8>,
            signature: Vec<u8>,
            pubkey_sec1: Vec<u8>,
        }
        let sk = test_key(0x7A);
        let payload =
            bincode::serialize(&WorkerMessage::EpochClose { epoch: 1 }).expect("serialize msg");
        let sig: Signature = sk.sign(&payload); // v1 signed bare payload
        let v1 = WireEnvelopeV1 {
            sender_addr: address_from_signing_key(&sk).0,
            payload,
            signature: sig.to_der().as_bytes().to_vec(),
            pubkey_sec1: sk
                .verifying_key()
                .to_encoded_point(false)
                .as_bytes()
                .to_vec(),
        };
        let bytes = bincode::serialize(&v1).expect("ser v1");
        assert!(
            verify_with(&bytes, &mut fresh_guard()).is_err(),
            "v1 envelope (no signed context) must be rejected"
        );
    }

    /// SECREM-02 6.3 (FUA-COMPUTE-POOL-02): inner sender-identity
    /// fields must match the verified envelope signer.
    #[test]
    fn sender_binding_covers_step_commits_and_activations() {
        let me = WorkerAddress::repeat_byte(0x01);
        let other = WorkerAddress::repeat_byte(0x02);

        // StepCommitted binding (COMPUTE_POOL-001, retained).
        let ok = WorkerMessage::StepCommitted(step_commit_for(0, 0, me));
        assert!(enforce_sender_binding(me, &ok).is_ok());
        let forged = WorkerMessage::StepCommitted(step_commit_for(0, 0, other));
        assert!(enforce_sender_binding(me, &forged).is_err());

        // PipelineActivation binding (FUA-COMPUTE-POOL-02).
        let ok = WorkerMessage::PipelineActivation {
            request_id: 1,
            from_stage: 0,
            to_stage: 1,
            from_worker: me,
            payload: vec![1, 2, 3],
        };
        assert!(enforce_sender_binding(me, &ok).is_ok());
        let forged = WorkerMessage::PipelineActivation {
            request_id: 1,
            from_stage: 0,
            to_stage: 1,
            from_worker: other,
            payload: vec![1, 2, 3],
        };
        assert!(enforce_sender_binding(me, &forged).is_err());

        // Non-identity-bearing messages pass through.
        assert!(enforce_sender_binding(me, &WorkerMessage::EpochClose { epoch: 3 }).is_ok());
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

        // Attacker signs their own envelope (valid v2 context) but
        // stamps `sender_addr` with the legitimate worker's address.
        let mut forged = signed_envelope(
            &attacker_sk,
            &WorkerMessage::EpochClose { epoch: 0 },
            test_ctx(42),
        );
        forged.sender_addr = legit_addr.0; // lie: pretending to be legit worker
        let forged_bytes = bincode::serialize(&forged).expect("ser forged");

        // The verify path must reject because `sender_addr` is
        // derived from the embedded pubkey, and the embedded
        // pubkey is the attacker's, which does not hash to the
        // legitimate worker's address.
        let err = verify_with(&forged_bytes, &mut fresh_guard())
            .expect_err("impersonation attempt must fail verification");
        assert!(
            err.contains("sender_addr mismatch"),
            "unexpected rejection reason: {err}"
        );
    }

    /// SECREM-02 6.3 RED→GREEN (CITRATE_COMPUTE_POOL-2026-05-31-002):
    /// a byte-identical envelope replayed a second time must be
    /// rejected (nonce already seen). Pre-fix `verify_envelope` was
    /// stateless and accepted the replay (red run recorded in the
    /// remediation log).
    #[tokio::test]
    async fn replayed_envelope_is_rejected_on_second_delivery() {
        let sk = test_key(0x5A);
        let env = signed_envelope(
            &sk,
            &WorkerMessage::StepAdvance { epoch: 0, step: 2 },
            test_ctx(77),
        );
        let bytes = bincode::serialize(&env).expect("ser");
        let mut guard = fresh_guard();
        assert!(
            verify_with(&bytes, &mut guard).is_ok(),
            "first delivery verifies"
        );
        let err = verify_with(&bytes, &mut guard)
            .expect_err("replayed envelope (same sender+nonce) must be rejected");
        assert!(err.contains("replayed"), "err was: {err}");

        // A DIFFERENT nonce from the same sender is still accepted.
        let env2 = signed_envelope(
            &sk,
            &WorkerMessage::StepAdvance { epoch: 0, step: 3 },
            test_ctx(78),
        );
        let bytes2 = bincode::serialize(&env2).expect("ser");
        assert!(
            verify_with(&bytes2, &mut guard).is_ok(),
            "fresh nonce from same sender accepted"
        );
    }

    #[tokio::test]
    async fn register_and_recv_without_messages_blocks() {
        let sk = test_key(0x4A);
        let transport = LibP2pTransport::new(sk, listen_addr(), vec![], test_scope())
            .await
            .expect("starts");
        let _ = wait_listener(&transport).await;

        let peer = WorkerAddress::repeat_byte(0xFE);
        transport.register(peer).await;

        // recv should block (no messages, no peers). Poll briefly.
        let res = tokio::time::timeout(Duration::from_millis(200), transport.recv(peer)).await;
        assert!(res.is_err(), "recv must block when no messages queued");
    }
}
