//! citrate-training-worker — DataParallel training worker daemon
//! for CM-07.
//!
//! # What this crate does
//!
//! A training worker is one provider node in a DataParallel pool.
//! It:
//!
//!   1. Subscribes to a training job's on-chain events
//!   2. Joins the job by posting stake
//!   3. Loads starting weights (identical across all workers)
//!   4. Loops per epoch, per step:
//!      - Forward + backward on local data shard
//!      - Ring-all-reduces quantized gradient with peers
//!      - Applies averaged gradient
//!      - Computes step commitment per ADR-008
//!      - Gossips step commitment to the coordinator
//!   5. When elected coordinator for an epoch, aggregates every
//!      worker's step commitments into a Merkle tree and posts the
//!      epoch root on-chain via `commitEpoch`.
//!   6. After all epochs close and the challenge window passes,
//!      receives accumulated payment + stake return at
//!      `finalizeTrainingJob`.
//!
//! # Stage scope
//!
//! Per the CM-07/08 staged execution plan:
//!
//! - **S0 (this slice)**: in-process `InProcessTransport` +
//!   `DeterministicTinyModel` + `MockChainClient`. No network, no
//!   GPU, no live chain. Validates the state machine end-to-end
//!   with 3 workers × 2 epochs × 2 steps.
//! - **S1**: `LibP2pTransport` over LAN. Still tiny model, still
//!   mock chain. Validates network protocol.
//! - **S2**: `TchGpuBackend` + real chain (Citrate testnet) + real
//!   dataset via IPFS. Validates GPU numerical stability.
//! - **S3**: pilot cluster.
//!
//! # Safe mocks (MOCKS.md entries)
//!
//! Behind-trait substitutions, each with an S1+ replacement:
//!
//! | Trait | S0 impl | S1+ replacement |
//! |-------|---------|-----------------|
//! | `ModelBackend` | `DeterministicTinyModel` | `TchGpuBackend` (S2) |
//! | `Transport` | `InProcessTransport` | `LibP2pTransport` (S1) |
//! | `ChainClient` | `MockChainClient` | `HttpChainClient` (S1) |

pub mod attestation;
pub mod backend;
#[cfg(feature = "candle-gpu")]
pub mod candle_backend;
pub mod chain;
pub mod events;
pub mod http_chain_pipeline;
pub mod job_artifacts;
/// The real NAT-backed training backend. Feature-gated because it pulls Candle;
/// the artifact-verification and commitment layers it depends on are NOT gated,
/// so the honesty checks compile and test everywhere.
#[cfg(feature = "nat")]
pub mod nat_backend;
pub mod q16_commitment;
pub mod zone_delta;
pub mod http_chain_training;
pub mod libp2p_transport;
pub mod merkle;
pub mod pipeline;
pub mod quantize;
pub mod transport;
pub mod types;
pub mod wallet;
pub mod worker;

pub use backend::{DeterministicTinyModel, MaliciousTinyModel, ModelBackend, Tensor};
pub use chain::{ChainClient, ChainError, JobChainSnapshot, MockChainClient};
pub use merkle::{compute_epoch_root, compute_leaf};
pub use quantize::{quantize_tensor, QuantizedGradient};
pub use libp2p_transport::{LibP2pTransport, LibP2pTransportError};
pub use transport::{InProcessTransport, Transport, WorkerMessage};
pub use types::{
    B256, CommitmentHash, EpochIndex, JobId, PrevWeightsHash, StepCommit, StepIndex,
    TrainingJobSpec, WeightsHash, WorkerAddress,
};
pub use worker::{Worker, WorkerConfig, WorkerOutcome};

pub use attestation::{
    build_submit_strict_bound_call, encode_submit_strict_bound_calldata, parse_jwt, reattest_at,
    validate_registry_address, AttestationBundle, AttestationError, FixtureAttestationSource,
    ParsedJwt,
};
pub use events::{classify, EventKind, RawLog};
pub use job_artifacts::{Architecture, ArtifactError, ArtifactStore, CommitmentGrid, JobArtifacts};
pub use q16_commitment::{q16_step_commitment, q16_tensor_commitment, to_q16, Q16Tensor, Q16_ONE};
pub use zone_delta::{zone_deltas, zone_of, DeltaError, ZoneDelta, SHARED};
pub use http_chain_pipeline::HttpPipelineChainClient;
pub use http_chain_training::HttpChainClient;
pub use wallet::{tx_hash_of_signed, Eip1559Tx, Wallet, WalletError};

pub use pipeline::{
    pipeline_stage_forward, MockPipelineChainClient, PipelineChainClient, PipelineChainError,
    PipelineJobId, PipelineJobSpec, PipelineRequestId, PipelineRequestSnapshot,
    PipelineRequestState, PipelineWorker, StageIndex, StageRole,
};
