//! Gradient quantization per ADR-008.
//!
//! Each gradient tensor is quantized to int8 per component with a
//! single float32 scale factor. The scale is `max(|G|) / 127`; the
//! per-component q = round(G / scale), clamped to [-128, 127].
//!
//! The output `QPACK = scale_bytes (4 bytes BE float32) ||
//! quantized_values (H × W int8)` is hashed with keccak256 to
//! produce the per-tensor commitment.
//!
//! Step commitment: `keccak256(tensor_commit_0 || tensor_commit_1
//! || ...)` across all tensors in canonical layer order.

use sha3::{Digest, Keccak256};

use crate::types::B256;

/// A quantized gradient tensor per ADR-008.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedGradient {
    /// Float32 scale factor. Multiplying a q-value by this gives
    /// back the original (dequantized) gradient component.
    pub scale: f32,
    /// Flattened int8 quantized values. Row-major layout.
    pub q_values: Vec<i8>,
}

/// Quantize a flat slice of f32 gradient values into int8.
///
/// Returns a `QuantizedGradient` carrying both the scale and the
/// quantized values. Per ADR-008, `scale = max(|g|) / 127`. A
/// zero-magnitude tensor yields scale = 0 and all-zero quantized
/// values (safe degenerate case).
pub fn quantize_tensor(grad: &[f32]) -> QuantizedGradient {
    let max_abs = grad.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = max_abs / 127.0;
    let q_values: Vec<i8> = if scale == 0.0 {
        vec![0; grad.len()]
    } else {
        grad.iter()
            .map(|&g| {
                let q = (g / scale).round();
                q.clamp(-128.0, 127.0) as i8
            })
            .collect()
    };
    QuantizedGradient { scale, q_values }
}

/// Dequantize for verification purposes (challenger side). Reverses
/// the quantization; the result is within `scale` of the original
/// per component.
pub fn dequantize_tensor(q: &QuantizedGradient) -> Vec<f32> {
    q.q_values.iter().map(|&v| (v as f32) * q.scale).collect()
}

/// Per-tensor commitment hash: keccak256(scale_bytes (BE f32) ||
/// q_values as bytes).
pub fn tensor_commitment(q: &QuantizedGradient) -> B256 {
    let mut hasher = Keccak256::new();
    hasher.update(q.scale.to_be_bytes());
    // int8 -> u8 byte-identical reinterpret via as_ptr cast; use
    // the safe `iter().copied() as u8` since signed/unsigned share
    // bit representations.
    let bytes: Vec<u8> = q.q_values.iter().map(|&v| v as u8).collect();
    hasher.update(&bytes);
    let out = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    B256::from(h)
}

/// Step commitment: keccak256 over concatenated per-tensor
/// commitments in canonical layer order. Layer order is fixed by
/// the model's starting weights manifest; workers agree on the
/// order at job start (out-of-band, via the model's schema).
pub fn step_commitment(tensor_commits: &[B256]) -> B256 {
    let mut hasher = Keccak256::new();
    for tc in tensor_commits {
        hasher.update(tc.as_bytes());
    }
    let out = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    B256::from(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_tensor_produces_zero_scale() {
        let q = quantize_tensor(&[0.0, 0.0, 0.0]);
        assert_eq!(q.scale, 0.0);
        assert_eq!(q.q_values, vec![0, 0, 0]);
    }

    #[test]
    fn max_component_maps_to_127() {
        let q = quantize_tensor(&[1.0, -1.0, 0.5]);
        // scale = 1.0 / 127, so 1.0 → 127, -1.0 → -127, 0.5 → ~63
        assert_eq!(q.q_values[0], 127);
        assert_eq!(q.q_values[1], -127);
        assert!((q.q_values[2] as i32 - 63).abs() <= 1);
    }

    #[test]
    fn dequantize_round_trips_within_scale() {
        let grad = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        let q = quantize_tensor(&grad);
        let dq = dequantize_tensor(&q);
        for (original, recovered) in grad.iter().zip(dq.iter()) {
            assert!(
                (original - recovered).abs() <= q.scale,
                "dequant error {} exceeds scale {}",
                (original - recovered).abs(),
                q.scale
            );
        }
    }

    #[test]
    fn tensor_commitment_is_deterministic() {
        let grad = vec![0.1, -0.2, 0.3];
        let q1 = quantize_tensor(&grad);
        let q2 = quantize_tensor(&grad);
        assert_eq!(tensor_commitment(&q1), tensor_commitment(&q2));
    }

    #[test]
    fn tensor_commitment_changes_with_values() {
        let q1 = quantize_tensor(&[0.1, 0.2, 0.3]);
        let q2 = quantize_tensor(&[0.1, 0.2, 0.4]);
        assert_ne!(tensor_commitment(&q1), tensor_commitment(&q2));
    }

    #[test]
    fn step_commitment_aggregates_tensors() {
        let tc1 = tensor_commitment(&quantize_tensor(&[0.1, 0.2]));
        let tc2 = tensor_commitment(&quantize_tensor(&[0.3, 0.4]));
        let single = step_commitment(&[tc1]);
        let both = step_commitment(&[tc1, tc2]);
        assert_ne!(single, both);
    }
}
