//! Core types shared across the worker, coordinator, and chain
//! client. Mirror the on-chain `ComputePoolTraining.sol` struct
//! shapes where applicable.

use ethereum_types::{Address, H256};
use serde::{Deserialize, Serialize};

/// 32-byte hash (keccak256). Re-export of ethereum_types::H256 so
/// callers don't need a direct dep.
pub type B256 = H256;

/// 20-byte EVM worker address.
pub type WorkerAddress = Address;

/// Sequential job identifier assigned by ComputePoolTraining at
/// `requestTrainingJob`. Opaque numeric handle.
pub type JobId = u64;

/// Epoch index, zero-based. Epochs execute strictly sequentially
/// (see DataParallelTrainingJob.tla EpochMonotonic invariant).
pub type EpochIndex = u32;

/// Step index within an epoch, zero-based. Resets to 0 at epoch
/// boundary.
pub type StepIndex = u32;

/// Hash of the model weights at a point in training. Keccak256 over
/// canonical tensor layout. Also used as the "step link" from step
/// s to step s+1 — the output of step s is the input of step s+1.
pub type WeightsHash = B256;

/// Hash of the prior step's post-weights, the input to this step's
/// forward pass. Distinct from the step commitment itself; serves as
/// the chain link between step commits.
pub type PrevWeightsHash = B256;

/// Step commitment hash per ADR-008: keccak256 over the quantized
/// gradient tensor commitments produced by this step.
pub type CommitmentHash = B256;

/// A step's on-chain-worthy commitment envelope. This is the thing
/// workers gossip to the coordinator (and to challengers via the
/// libp2p mesh archive). The coordinator aggregates these into a
/// Merkle tree; only the tree root hits the chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepCommit {
    pub epoch: EpochIndex,
    pub step: StepIndex,
    pub worker: WorkerAddress,
    pub commitment: CommitmentHash,
    pub prev_weights: PrevWeightsHash,
}

/// Training-job spec as presented to the worker. Mirrors
/// ComputePoolTraining.TrainingJobSpec but with Rust-native types.
#[derive(Clone, Debug)]
pub struct TrainingJobSpec {
    pub model_start_hash: WeightsHash,
    pub dataset_hash: B256,
    pub epoch_count: u32,
    pub steps_per_epoch: u32,
    pub min_workers: u32,
    pub max_workers: u32,
    pub challenge_window_blocks: u32,
    pub per_epoch_budget: u128,
    pub per_worker_stake: u128,
}
