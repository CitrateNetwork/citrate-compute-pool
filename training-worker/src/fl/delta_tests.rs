use super::*;
use crate::fl::gguf::{encode_f32, OutTensor, Value};

pub(crate) fn adapter_kv() -> Vec<(String, Value)> {
    vec![
        ("general.architecture".into(), Value::Str("gemma4".into())),
        ("general.type".into(), Value::Str("adapter".into())),
        ("adapter.type".into(), Value::Str("lora".into())),
        ("adapter.lora.alpha".into(), Value::F32(16.0)),
    ]
}

/// Write an adapter whose tensors are deliberately NOT in name order, load it.
pub(crate) fn write_adapter(dir: &std::path::Path, file: &str, b_vals: &[f32], a_vals: &[f32]) -> Adapter {
    std::fs::create_dir_all(dir).expect("dir");
    let p = dir.join(file);
    let bytes = encode_f32(
        &adapter_kv(),
        &[
            OutTensor {
                name: "blk.0.attn_q.weight.lora_b",
                dims: &[2, 3],
                values: b_vals,
            },
            OutTensor {
                name: "blk.0.attn_q.weight.lora_a",
                dims: &[4, 2],
                values: a_vals,
            },
        ],
    )
    .expect("encode");
    std::fs::write(&p, bytes).expect("write");
    Adapter::load(&p).expect("load")
}

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("fl-delta-{tag}-{}", std::process::id()))
}

const A0: [f32; 8] = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08];
const B0: [f32; 6] = [0.0; 6];

#[test]
fn the_delta_follows_name_order_and_the_q16_grid() {
    let dir = tmp("order");
    let start = write_adapter(&dir, "s.gguf", &B0, &A0);
    let mut a1 = A0;
    a1[0] += 0.5;
    let b1 = [0.25f32, 0.0, -0.25, 0.0, 0.0, 1.0];
    let trained = write_adapter(&dir, "t.gguf", &b1, &a1);
    let d = compute(&start, &trained, 0).expect("delta");
    // lora_a sorts before lora_b: eight a-values first, then six b-values.
    assert_eq!(d.len(), 14);
    // 0.5 on the Q16 grid; the other a-values did not move.
    assert_eq!(d[0], 32768);
    assert!(d[1..8].iter().all(|v| *v == 0));
    assert_eq!(&d[8..], &[16384, 0, -16384, 0, 0, 65536]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn scaling_keeps_small_updates_on_the_grid() {
    let dir = tmp("scale");
    let start = write_adapter(&dir, "s.gguf", &B0, &A0);
    let mut b1 = B0;
    b1[0] = 1.0e-5; // below one Q16 unit (1.5e-5) unscaled
    let trained = write_adapter(&dir, "t.gguf", &b1, &A0);
    let unscaled = compute(&start, &trained, 0).expect("delta");
    let scaled = compute(&start, &trained, 8).expect("delta");
    assert_eq!(unscaled[8], 1); // 0.655 rounds to 1
    assert_eq!(scaled[8], 168); // 1e-5 * 256 * 65536 = 167.77
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn mismatched_and_broken_adapters_are_refused() {
    let dir = tmp("refuse");
    let start = write_adapter(&dir, "s.gguf", &B0, &A0);
    let mut b1 = B0;
    b1[2] = f32::NAN;
    let nan = write_adapter(&dir, "n.gguf", &b1, &A0);
    assert!(matches!(
        compute(&start, &nan, 8),
        Err(DeltaError::NotFinite { .. })
    ));
    let mut b2 = B0;
    b2[1] = 1.0e9;
    let huge = write_adapter(&dir, "h.gguf", &b2, &A0);
    assert!(matches!(
        compute(&start, &huge, 8),
        Err(DeltaError::OutOfRange { .. })
    ));

    // A different shape is a different model.
    let p = dir.join("other.gguf");
    let bytes = encode_f32(
        &adapter_kv(),
        &[OutTensor {
            name: "blk.0.attn_q.weight.lora_a",
            dims: &[8],
            values: &A0,
        }],
    )
    .expect("encode");
    std::fs::write(&p, bytes).expect("write");
    let other = Adapter::load(&p).expect("load");
    assert!(matches!(
        compute(&start, &other, 8),
        Err(DeltaError::Mismatch(_))
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn merging_a_lone_participants_delta_recovers_its_adapter() {
    let dir = tmp("merge");
    let start = write_adapter(&dir, "s.gguf", &B0, &A0);
    let b1 = [0.125f32, -0.5, 0.0, 0.75, 0.001, -0.002];
    let trained = write_adapter(&dir, "t.gguf", &b1, &A0);
    let d = compute(&start, &trained, 8).expect("delta");
    let merged = merge(&start, &d, 8).expect("merge");
    // Back in file order: lora_b first.
    for (got, want) in merged[0].iter().zip(b1.iter()) {
        assert!((got - want).abs() < 1.0e-7, "{got} vs {want}");
    }
    assert_eq!(merged[1], A0.to_vec());
    // A zero aggregate is the start adapter itself.
    let z = merge(&start, &vec![0; d.len()], 8).expect("merge");
    assert_eq!(z, start.values);
    // The merged file is a loadable adapter with the start's metadata.
    let bytes = merged_gguf(&start, &merged).expect("gguf");
    let p = dir.join("m.gguf");
    std::fs::write(&p, bytes).expect("write");
    let m = Adapter::load(&p).expect("load merged");
    assert_eq!(m.header.kv, start.header.kv);
    assert_eq!(m.values, merged);
    assert!(merge(&start, &[0; 3], 8).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rows_and_roots_commit_to_every_value() {
    let values: Vec<i64> = (0..10).map(|i| i * 3 - 7).collect();
    let rows = row_hashes(&values, 4).expect("rows");
    assert_eq!(rows.len(), 3);
    assert_eq!(row(&values, 4, 2), &values[8..10]);
    let mut bytes = Vec::new();
    for v in &values[4..8] {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    assert_eq!(rows[1], keccak(&[&bytes]));
    let root = delta_root(&values, 4).expect("root");
    for i in 0..values.len() {
        let mut w = values.clone();
        w[i] += 1;
        assert_ne!(delta_root(&w, 4).expect("root"), root);
    }
    assert_eq!(row_hashes(&values, 0), Err(DeltaError::ChunkDim));
}

fn sample_artifact() -> Artifact {
    Artifact {
        value_scale_log2: 8,
        round_id: [1; 32],
        worker: [2; 20],
        start_adapter_sha256: [3; 32],
        trained_adapter_sha256: [4; 32],
        manifest_hash: [5; 32],
        chunk_dim: 256,
        values: vec![1, -2, i64::MAX, i64::MIN, 0],
    }
}

#[test]
fn an_artifact_round_trips() {
    let a = sample_artifact();
    let bytes = a.encode();
    assert_eq!(bytes.len(), HEADER_LEN + 5 * 8);
    assert_eq!(Artifact::decode(&bytes, 5).expect("decode"), a);
}

#[test]
fn malformed_artifacts_are_refused() {
    let bytes = sample_artifact().encode();
    assert!(matches!(
        Artifact::decode(&bytes, 4),
        Err(DeltaError::TooLarge(5, 4))
    ));
    assert!(Artifact::decode(&bytes[..bytes.len() - 1], 5).is_err());
    assert!(Artifact::decode(&bytes[..100], 5).is_err());
    let mut b = bytes.clone();
    b[0] = b'X';
    assert!(Artifact::decode(&b, 5).is_err());
    let mut b = bytes.clone();
    b[5] = 2;
    assert!(Artifact::decode(&b, 5).is_err());
    let mut b = bytes.clone();
    b[7] = 1;
    assert!(Artifact::decode(&b, 5).is_err());
    let mut b = bytes.clone();
    b[156..160].copy_from_slice(&0u32.to_be_bytes());
    assert_eq!(Artifact::decode(&b, 5), Err(DeltaError::ChunkDim));
    // A count that would overflow the allocation is refused by the limit.
    let mut b = bytes;
    b[160..168].copy_from_slice(&u64::MAX.to_be_bytes());
    assert!(Artifact::decode(&b, u64::MAX).is_err());
}

#[test]
fn the_manifest_pins_names_and_shapes() {
    let dir = tmp("manifest");
    let s = write_adapter(&dir, "s.gguf", &B0, &A0);
    let t = write_adapter(&dir, "t.gguf", &[1.0; 6], &A0);
    // Values do not change the layout.
    assert_eq!(manifest_hash(&s), manifest_hash(&t));
    let p = dir.join("o.gguf");
    let bytes = encode_f32(
        &adapter_kv(),
        &[
            OutTensor {
                name: "blk.0.attn_q.weight.lora_b",
                dims: &[3, 2],
                values: &B0,
            },
            OutTensor {
                name: "blk.0.attn_q.weight.lora_a",
                dims: &[4, 2],
                values: &A0,
            },
        ],
    )
    .expect("encode");
    std::fs::write(&p, bytes).expect("write");
    let o = Adapter::load(&p).expect("load");
    assert_ne!(manifest_hash(&s), manifest_hash(&o));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_header_only_artifact_decodes_to_no_values() {
    let mut a = sample_artifact();
    a.values.clear();
    let bytes = a.encode();
    assert_eq!(bytes.len(), HEADER_LEN);
    assert_eq!(Artifact::decode(&bytes, 0).expect("decode"), a);
}
