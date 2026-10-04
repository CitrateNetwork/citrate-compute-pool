use super::*;

/// The golden chunk shared with citrate-chain (`tools/fl-replay/tests/golden.rs`)
/// and the ledger's forge tests: three participants, four coordinates.
pub(crate) fn golden_rows() -> [[i64; 4]; 3] {
    [
        [65536, -32768, 0, 100],
        [32768, -32768, 0, 50],
        [-16384, 16384, 0, 25],
    ]
}

pub(crate) fn golden_rules() -> ChunkRules {
    ChunkRules {
        confidence: ConfidenceRule::Nonzero,
        weight: WeightRule::Uniform,
        threshold_pos: 32768,
        threshold_neg: -32768,
    }
}

/// What the chain kernel returns for the golden chunk (pinned on the chain side
/// too, by running the precompile's own `aggregate`).
pub(crate) const GOLDEN_OUTPUT_HEX: &str = "0000000000006aa9ffffffffffffbfff00000000000000000000000000000039\
03030001";

#[test]
fn the_golden_input_is_byte_exact() {
    let rows = golden_rows();
    let refs: Vec<&[i64]> = rows.iter().map(|r| r.as_slice()).collect();
    let got = chunk_input(&refs, &golden_rules()).expect("input");

    let mut want = Vec::new();
    want.extend_from_slice(&4u32.to_be_bytes());
    want.extend_from_slice(&3u32.to_be_bytes());
    for r in &rows {
        for v in r {
            want.extend_from_slice(&v.to_be_bytes());
        }
    }
    for r in &rows {
        for v in r {
            let c: i64 = if *v == 0 { 0 } else { 65536 };
            want.extend_from_slice(&c.to_be_bytes());
        }
    }
    for _ in 0..3 {
        want.extend_from_slice(&21845i64.to_be_bytes());
    }
    want.extend_from_slice(&32768i64.to_be_bytes());
    want.extend_from_slice(&(-32768i64).to_be_bytes());
    assert_eq!(got, want);
    assert_eq!(got.len(), 24 + 16 * 3 * 4 + 8 * 3);
}

#[test]
fn the_golden_output_decodes() {
    let bytes = hex::decode(GOLDEN_OUTPUT_HEX).expect("hex");
    let out = parse_output(&bytes, 4).expect("parse");
    assert_eq!(out.values, vec![27305, -16385, 0, 57]);
    assert_eq!(out.states, vec![3, 3, 0, 1]);
}

#[test]
fn uniform_weights_floor_on_the_grid() {
    assert_eq!(WeightRule::Uniform.weights(1), vec![65536]);
    assert_eq!(WeightRule::Uniform.weights(3), vec![21845; 3]);
    assert_eq!(WeightRule::Uniform.weights(4), vec![16384; 4]);
}

#[test]
fn shapes_outside_the_precompile_caps_are_refused() {
    let r = golden_rules();
    assert_eq!(chunk_input(&[], &r), Err(BelnapWireError::Participants(0)));
    let empty: &[i64] = &[];
    assert_eq!(chunk_input(&[empty], &r), Err(BelnapWireError::Dim(0)));
    let wide = vec![1i64; MAX_DIM + 1];
    assert_eq!(
        chunk_input(&[wide.as_slice()], &r),
        Err(BelnapWireError::Dim(MAX_DIM + 1))
    );
    let a = [1i64, 2];
    let b = [1i64];
    assert_eq!(
        chunk_input(&[&a, &b], &r),
        Err(BelnapWireError::Ragged {
            index: 1,
            got: 1,
            dim: 2
        })
    );
}

#[test]
fn malformed_outputs_are_refused() {
    assert!(matches!(
        parse_output(&[0u8; 17], 2),
        Err(BelnapWireError::OutputLength { .. })
    ));
    let mut bad = vec![0u8; 9];
    bad[8] = 4;
    assert_eq!(parse_output(&bad, 1), Err(BelnapWireError::State(4)));
    // A codeless address answers a call with empty data; that is never an
    // aggregate.
    assert!(parse_output(&[], 1).is_err());
}

#[test]
fn the_precompile_caps_are_inclusive() {
    let r = golden_rules();
    let one = [7i64];
    let rows = vec![one.as_slice(); MAX_N];
    let input = chunk_input(&rows, &r).expect("MAX_N participants are allowed");
    assert_eq!(u32::from_be_bytes([input[4], input[5], input[6], input[7]]) as usize, MAX_N);
    let over = vec![one.as_slice(); MAX_N + 1];
    assert_eq!(chunk_input(&over, &r), Err(BelnapWireError::Participants(MAX_N + 1)));
    let wide = vec![1i64; MAX_DIM];
    assert!(chunk_input(&[wide.as_slice()], &r).is_ok(), "MAX_DIM is allowed");
}
