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

pub mod backend;
pub mod chain;
pub mod merkle;
pub mod quantize;
pub mod transport;
pub mod types;
pub mod worker;

pub use backend::{DeterministicTinyModel, ModelBackend, Tensor};
pub use chain::{ChainClient, ChainError, JobChainSnapshot, MockChainClient};
pub use merkle::{compute_epoch_root, compute_leaf};
pub use quantize::{quantize_tensor, QuantizedGradient};
pub use transport::{InProcessTransport, Transport, WorkerMessage};
pub use types::{
    B256, CommitmentHash, EpochIndex, JobId, PrevWeightsHash, StepCommit, StepIndex,
    TrainingJobSpec, WeightsHash, WorkerAddress,
};
pub use worker::{Worker, WorkerConfig, WorkerOutcome};
