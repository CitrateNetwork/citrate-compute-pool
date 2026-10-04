use super::*;
use std::io::Cursor as IoCursor;

fn adapter_kv() -> Vec<(String, Value)> {
    vec![
        ("general.architecture".into(), Value::Str("gemma4".into())),
        ("general.type".into(), Value::Str("adapter".into())),
        ("adapter.type".into(), Value::Str("lora".into())),
        ("adapter.lora.alpha".into(), Value::F32(16.0)),
        (
            "test.array".into(),
            Value::Array(4, vec![Value::U32(1), Value::U32(2)]),
        ),
    ]
}

#[test]
fn an_encoded_adapter_reads_back_identically() {
    let a = [0.5f32, -1.25, 3.0, 0.0, 7.5, -0.125];
    let b = [1.0f32, 2.0, 3.0, 4.0];
    let bytes = encode_f32(
        &adapter_kv(),
        &[
            OutTensor {
                name: "blk.0.attn_q.weight.lora_a",
                dims: &[3, 2],
                values: &a,
            },
            OutTensor {
                name: "blk.0.attn_q.weight.lora_b",
                dims: &[2, 2],
                values: &b,
            },
        ],
    )
    .expect("encode");
    let mut r = IoCursor::new(bytes);
    let g = read_header(&mut r).expect("header");
    assert_eq!(g.version, 3);
    assert_eq!(g.kv, adapter_kv());
    assert!(g.is_lora_adapter());
    assert_eq!(g.tensors.len(), 2);
    assert_eq!(g.tensors[0].dims, vec![3, 2]);
    assert_eq!(g.data_offset % 32, 0);
    let ta = g.tensors[0].clone();
    let tb = g.tensors[1].clone();
    assert_eq!(read_tensor_f32(&mut r, &g, &ta).expect("a"), a.to_vec());
    assert_eq!(read_tensor_f32(&mut r, &g, &tb).expect("b"), b.to_vec());
}

#[test]
fn f16_conversion_is_exact_on_known_values() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    assert_eq!(f16_to_f32(0x7bff), 65504.0);
    // Smallest subnormal: 2^-24.
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    // Largest subnormal.
    assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert!(f16_to_f32(0x7c00).is_infinite());
    assert!(f16_to_f32(0x7e00).is_nan());
}

#[test]
fn every_f16_matches_the_value_formula() {
    for h in 0u16..=u16::MAX {
        let exp = (h >> 10) & 0x1f;
        if exp == 0x1f {
            continue;
        }
        let man = f64::from(h & 0x3ff);
        let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
        let v = if exp == 0 {
            sign * man * 2f64.powi(-24)
        } else {
            sign * (1.0 + man / 1024.0) * 2f64.powi(i32::from(exp) - 15)
        };
        assert_eq!(f64::from(f16_to_f32(h)), v, "h={h:#06x}");
    }
}

#[test]
fn f16_and_bf16_tensors_read_as_f32() {
    // Build by hand: one F16 tensor [1.0, -2.0] and one BF16 tensor [1.5].
    let mut f = Vec::new();
    f.extend_from_slice(b"GGUF");
    f.extend_from_slice(&3u32.to_le_bytes());
    f.extend_from_slice(&2u64.to_le_bytes());
    f.extend_from_slice(&0u64.to_le_bytes());
    for (name, ty, off) in [("h", GGML_TYPE_F16, 0u64), ("b", GGML_TYPE_BF16, 32)] {
        f.extend_from_slice(&(name.len() as u64).to_le_bytes());
        f.extend_from_slice(name.as_bytes());
        f.extend_from_slice(&1u32.to_le_bytes());
        f.extend_from_slice(&(if name == "h" { 2u64 } else { 1 }).to_le_bytes());
        f.extend_from_slice(&ty.to_le_bytes());
        f.extend_from_slice(&off.to_le_bytes());
    }
    let start = f.len().div_ceil(32) * 32;
    f.resize(start, 0);
    f.extend_from_slice(&0x3c00u16.to_le_bytes());
    f.extend_from_slice(&0xc000u16.to_le_bytes());
    f.resize(start + 32, 0);
    f.extend_from_slice(&((1.5f32.to_bits() >> 16) as u16).to_le_bytes());
    let mut r = IoCursor::new(f);
    let g = read_header(&mut r).expect("header");
    let h = g.tensors[0].clone();
    let b = g.tensors[1].clone();
    assert_eq!(read_tensor_f32(&mut r, &g, &h).expect("f16"), vec![1.0, -2.0]);
    assert_eq!(read_tensor_f32(&mut r, &g, &b).expect("bf16"), vec![1.5]);
}

#[test]
fn a_quantized_tensor_is_refused_not_dequantized() {
    let mut f = Vec::new();
    f.extend_from_slice(b"GGUF");
    f.extend_from_slice(&3u32.to_le_bytes());
    f.extend_from_slice(&1u64.to_le_bytes());
    f.extend_from_slice(&0u64.to_le_bytes());
    f.extend_from_slice(&1u64.to_le_bytes());
    f.extend_from_slice(b"q");
    f.extend_from_slice(&1u32.to_le_bytes());
    f.extend_from_slice(&32u64.to_le_bytes());
    f.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
    f.extend_from_slice(&0u64.to_le_bytes());
    f.resize(f.len().div_ceil(32) * 32 + 64, 0);
    let mut r = IoCursor::new(f);
    let g = read_header(&mut r).expect("header");
    let t = g.tensors[0].clone();
    assert!(matches!(
        read_tensor_f32(&mut r, &g, &t),
        Err(GgufError::TensorType { ty: 8, .. })
    ));
}

#[test]
fn hostile_lengths_are_refused_before_allocating() {
    // A string claiming 2^40 bytes in a tiny file.
    let mut f = Vec::new();
    f.extend_from_slice(b"GGUF");
    f.extend_from_slice(&3u32.to_le_bytes());
    f.extend_from_slice(&0u64.to_le_bytes());
    f.extend_from_slice(&1u64.to_le_bytes());
    f.extend_from_slice(&(1u64 << 40).to_le_bytes());
    assert!(matches!(
        read_header(IoCursor::new(f)),
        Err(GgufError::Bounds(_))
    ));

    // An array claiming more elements than bytes remain.
    let mut f = Vec::new();
    f.extend_from_slice(b"GGUF");
    f.extend_from_slice(&3u32.to_le_bytes());
    f.extend_from_slice(&0u64.to_le_bytes());
    f.extend_from_slice(&1u64.to_le_bytes());
    f.extend_from_slice(&1u64.to_le_bytes());
    f.extend_from_slice(b"k");
    f.extend_from_slice(&9u32.to_le_bytes());
    f.extend_from_slice(&0u32.to_le_bytes());
    f.extend_from_slice(&1_000_000u64.to_le_bytes());
    assert!(matches!(
        read_header(IoCursor::new(f)),
        Err(GgufError::Bounds("array length"))
    ));

    // A tensor table claiming 2^30 tensors.
    let mut f = Vec::new();
    f.extend_from_slice(b"GGUF");
    f.extend_from_slice(&3u32.to_le_bytes());
    f.extend_from_slice(&(1u64 << 30).to_le_bytes());
    f.extend_from_slice(&0u64.to_le_bytes());
    assert!(matches!(
        read_header(IoCursor::new(f)),
        Err(GgufError::Bounds("tensor count"))
    ));
}

#[test]
fn bad_magic_and_versions_are_refused() {
    assert!(matches!(
        read_header(IoCursor::new(b"GGML\x03\0\0\0".to_vec())),
        Err(GgufError::BadMagic)
    ));
    let mut f = b"GGUF".to_vec();
    f.extend_from_slice(&9u32.to_le_bytes());
    f.extend_from_slice(&[0u8; 16]);
    assert!(matches!(
        read_header(IoCursor::new(f)),
        Err(GgufError::Version(9))
    ));
}

#[test]
fn a_tensor_past_the_end_of_the_file_is_refused() {
    let bytes = encode_f32(
        &adapter_kv(),
        &[OutTensor {
            name: "x.lora_a",
            dims: &[4],
            values: &[1.0, 2.0, 3.0, 4.0],
        }],
    )
    .expect("encode");
    let truncated = bytes[..bytes.len() - 4].to_vec();
    let mut r = IoCursor::new(truncated);
    let g = read_header(&mut r).expect("header");
    let t = g.tensors[0].clone();
    assert!(matches!(
        read_tensor_f32(&mut r, &g, &t),
        Err(GgufError::TensorBounds(_))
    ));
}

#[test]
fn duplicate_tensor_names_are_refused() {
    let bytes = encode_f32(
        &adapter_kv(),
        &[
            OutTensor {
                name: "x.lora_a",
                dims: &[1],
                values: &[1.0],
            },
            OutTensor {
                name: "x.lora_a",
                dims: &[1],
                values: &[2.0],
            },
        ],
    )
    .expect("encode");
    assert!(matches!(
        read_header(IoCursor::new(bytes)),
        Err(GgufError::Duplicate(_))
    ));
}

#[test]
fn a_model_that_is_not_an_adapter_is_refused_as_one() {
    let dir = std::env::temp_dir().join(format!("fl-gguf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let p = dir.join("model.gguf");
    let kv = vec![("general.architecture".to_string(), Value::Str("gemma4".into()))];
    let bytes = encode_f32(
        &kv,
        &[OutTensor {
            name: "blk.0.attn_q.weight",
            dims: &[2],
            values: &[1.0, 2.0],
        }],
    )
    .expect("encode");
    std::fs::write(&p, bytes).expect("write");
    assert!(matches!(Adapter::load(&p), Err(GgufError::Invalid(_))));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mismatched_dims_and_values_are_refused_at_encode() {
    assert!(encode_f32(
        &adapter_kv(),
        &[OutTensor {
            name: "x.lora_a",
            dims: &[3],
            values: &[1.0],
        }],
    )
    .is_err());
}
