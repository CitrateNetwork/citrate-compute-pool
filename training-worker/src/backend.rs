//! `ModelBackend` trait and the S0 deterministic implementation.
//!
//! The trait abstracts the GPU/CPU/training-runtime from the worker
//! state machine. For S0 we need a backend that is *deterministic*
//! (all workers produce the same commitment for the same step given
//! the same starting weights) but produces distinct outputs for
//! distinct steps (so the Merkle tree has >1 unique leaf and the
//! commitment flow is exercised end-to-end).
//!
//! `DeterministicTinyModel` satisfies both: its "forward + backward"
//! is a pure function of `(model_start_hash, epoch, step,
//! worker_shard_index)`. Different workers see different shards, so
//! their gradients differ per step; but every worker's computation
//! is reproducible, so a challenger can verify.
//!
//! Real GPU training (TchGpuBackend, CandleGpuBackend) slots in at
//! S2 without touching any other code in this crate — it's a
//! transparent substitution per the ADR-008 commitment scheme.

use async_trait::async_trait;
use sha3::{Digest, Keccak256};

use crate::quantize::{quantize_tensor, step_commitment, tensor_commitment};
use crate::types::{
    B256, CommitmentHash, EpochIndex, PrevWeightsHash, StepIndex, WeightsHash,
};

/// Simple flat-tensor abstraction. Production backends use
/// multi-dimensional tensor crates (tch, candle); for S0 a Vec<f32>
/// plus a shape metadata is sufficient.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    /// Canonical name/index of the tensor in the model's layer
    /// layout. Determines the order tensors enter step_commitment.
    pub layer_index: usize,
}

/// Result of a forward+backward pass: a set of gradient tensors
/// plus the new post-step weights hash.
#[derive(Clone, Debug)]
pub struct StepResult {
    pub gradients: Vec<Tensor>,
    pub post_weights_hash: WeightsHash,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Load starting weights for a job. For S0 the weights are
    /// conceptually "the identity" — the backend just returns the
    /// hash it was given; no tensor state is held. S2 real backends
    /// fetch from IPFS + load into GPU memory.
    async fn load_starting_weights(&self, model_start_hash: WeightsHash) -> anyhow::Result<WeightsHash>;

    /// Execute a forward + backward pass for a specific
    /// (epoch, step, worker_shard). Returns the gradient tensors
    /// for this step plus the post-weights hash.
    async fn forward_backward(
        &self,
        prev_weights: PrevWeightsHash,
        epoch: EpochIndex,
        step: StepIndex,
        worker_shard: u32,
    ) -> anyhow::Result<StepResult>;

    /// Compute the step commitment hash from the gradient tensors,
    /// per ADR-008. Default impl quantizes + hashes each tensor and
    /// concatenates in layer order.
    fn compute_step_commitment(&self, gradients: &[Tensor]) -> CommitmentHash {
        let mut ordered: Vec<&Tensor> = gradients.iter().collect();
        ordered.sort_by_key(|t| t.layer_index);
        let tensor_commits: Vec<B256> = ordered
            .iter()
            .map(|t| tensor_commitment(&quantize_tensor(&t.data)))
            .collect();
        step_commitment(&tensor_commits)
    }
}

/// S0 backend. Produces deterministic synthetic gradients from the
/// tuple (prev_weights, epoch, step, worker_shard) — no real ML,
/// but produces a valid commitment structure with the right
/// properties: deterministic, chainable (step s's post-weights
/// feed into step s+1), distinct per worker.
///
/// The "gradient" for each step is a 4-element f32 vector computed
/// from a keccak256 hash of the state tuple, producing values in
/// [-1, 1]. This is enough to exercise quantization, commitment
/// hashing, and Merkle aggregation end-to-end without any real
/// training.
pub struct DeterministicTinyModel;

impl DeterministicTinyModel {
    pub fn new() -> Self {
        Self
    }

    /// Derive synthetic gradient values from a state hash. Four
    /// f32 elements per tensor, one tensor per step. Bytes 0-3 →
    /// f32_1; bytes 4-7 → f32_2; etc. Each float is scaled to the
    /// range [-1, 1] for realistic quantization behavior.
    fn synthetic_gradient(state_hash: B256, layer_index: usize) -> Tensor {
        let bytes = state_hash.as_bytes();
        let mut data = Vec::with_capacity(4);
        for i in 0..4 {
            let offset = i * 4;
            let word = u32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            // Map to [-1, 1] deterministically.
            let f = (word as f32 / u32::MAX as f32) * 2.0 - 1.0;
            data.push(f);
        }
        Tensor { data, layer_index }
    }
}

impl Default for DeterministicTinyModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelBackend for DeterministicTinyModel {
    async fn load_starting_weights(
        &self,
        model_start_hash: WeightsHash,
    ) -> anyhow::Result<WeightsHash> {
        // S0: weights are abstract; we just trust the hash.
        Ok(model_start_hash)
    }

    async fn forward_backward(
        &self,
        prev_weights: PrevWeightsHash,
        epoch: EpochIndex,
        step: StepIndex,
        worker_shard: u32,
    ) -> anyhow::Result<StepResult> {
        // Derive two synthetic tensors from a state hash per the
        // doc comment. Two tensors keeps the step_commitment
        // structure non-trivial.
        let mut hasher = Keccak256::new();
        hasher.update(prev_weights.as_bytes());
        hasher.update(epoch.to_be_bytes());
        hasher.update(step.to_be_bytes());
        hasher.update(worker_shard.to_be_bytes());
        let state_hash_bytes = hasher.finalize();
        let mut sh = [0u8; 32];
        sh.copy_from_slice(&state_hash_bytes);
        let state_hash = B256::from(sh);

        let tensors = vec![
            Self::synthetic_gradient(state_hash, 0),
            Self::synthetic_gradient(state_hash, 1),
        ];

        // Post-weights: deterministic chain — hash of prev + tensor
        // values, so step s+1 depends on step s.
        let mut hasher = Keccak256::new();
        hasher.update(prev_weights.as_bytes());
        for t in &tensors {
            for v in &t.data {
                hasher.update(v.to_be_bytes());
            }
        }
        let post_bytes = hasher.finalize();
        let mut pb = [0u8; 32];
        pb.copy_from_slice(&post_bytes);
        let post_weights_hash = B256::from(pb);

        Ok(StepResult {
            gradients: tensors,
            post_weights_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_same_input_same_output() {
        let backend = DeterministicTinyModel::new();
        let prev = B256::repeat_byte(0xAA);
        let r1 = backend.forward_backward(prev, 0, 0, 0).await.unwrap();
        let r2 = backend.forward_backward(prev, 0, 0, 0).await.unwrap();
        assert_eq!(r1.post_weights_hash, r2.post_weights_hash);
        assert_eq!(
            backend.compute_step_commitment(&r1.gradients),
            backend.compute_step_commitment(&r2.gradients)
        );
    }

    #[tokio::test]
    async fn different_workers_produce_different_commits() {
        let backend = DeterministicTinyModel::new();
        let prev = B256::repeat_byte(0xAA);
        let r_w0 = backend.forward_backward(prev, 0, 0, 0).await.unwrap();
        let r_w1 = backend.forward_backward(prev, 0, 0, 1).await.unwrap();
        assert_ne!(
            backend.compute_step_commitment(&r_w0.gradients),
            backend.compute_step_commitment(&r_w1.gradients)
        );
    }

    #[tokio::test]
    async fn step_chain_is_non_trivial() {
        // Each step's post-weights differs so the NEXT step's input
        // is unique, preventing repeated-step counterfeiting.
        let backend = DeterministicTinyModel::new();
        let prev = B256::repeat_byte(0xAA);
        let r0 = backend.forward_backward(prev, 0, 0, 0).await.unwrap();
        let r1 = backend
            .forward_backward(r0.post_weights_hash, 0, 1, 0)
            .await
            .unwrap();
        assert_ne!(r0.post_weights_hash, r1.post_weights_hash);
    }
}
