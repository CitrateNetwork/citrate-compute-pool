use super::*;

pub(crate) fn sample_config() -> RoundConfig {
    RoundConfig {
        chain_id: 1337,
        ledger: [0x11; 20],
        cluster_id: [0x22; 32],
        base_model_sha256: [0x33; 32],
        start_adapter_sha256: [0x44; 32],
        roster: vec![[0x01; 20], [0x02; 20], [0x03; 20]],
        min_participants: 3,
        chunk_dim: 256,
        value_scale_log2: 8,
        threshold_pos: 32768,
        threshold_neg: -32768,
        confidence: ConfidenceRule::Nonzero,
        weight: WeightRule::Uniform,
        max_values: 1 << 26,
    }
}

#[test]
fn the_sample_config_is_valid() {
    sample_config().validate().expect("valid");
}

#[test]
fn every_rule_is_enforced() {
    let mut c = sample_config();
    c.roster = vec![[0x02; 20], [0x01; 20], [0x03; 20]];
    assert_eq!(c.validate(), Err(ConfigError::RosterOrder));

    let mut c = sample_config();
    c.roster = vec![[0x01; 20], [0x01; 20], [0x03; 20]];
    assert_eq!(c.validate(), Err(ConfigError::RosterOrder));

    let mut c = sample_config();
    c.min_participants = 2;
    assert_eq!(c.validate(), Err(ConfigError::MinParticipants(2)));

    let mut c = sample_config();
    c.min_participants = 4;
    assert!(matches!(
        c.validate(),
        Err(ConfigError::RosterTooSmall { .. })
    ));

    let mut c = sample_config();
    c.chunk_dim = 0;
    assert_eq!(c.validate(), Err(ConfigError::ChunkDim(0)));
    c.chunk_dim = 1025;
    assert_eq!(c.validate(), Err(ConfigError::ChunkDim(1025)));

    let mut c = sample_config();
    c.chunk_dim = 1024;
    c.roster = (1..=5u8).map(|b| [b; 20]).collect();
    assert!(matches!(c.validate(), Err(ConfigError::ChunkCells { .. })));

    let mut c = sample_config();
    c.value_scale_log2 = 17;
    assert_eq!(c.validate(), Err(ConfigError::Scale(17)));

    let mut c = sample_config();
    c.threshold_pos = 0;
    assert_eq!(c.validate(), Err(ConfigError::Thresholds));
    let mut c = sample_config();
    c.threshold_neg = 1;
    assert_eq!(c.validate(), Err(ConfigError::Thresholds));

    let mut c = sample_config();
    c.max_values = 0;
    assert_eq!(c.validate(), Err(ConfigError::MaxValues));

    let mut c = sample_config();
    c.cluster_id = [0; 32];
    assert_eq!(c.validate(), Err(ConfigError::Zero("cluster_id")));
    let mut c = sample_config();
    c.roster = vec![[0; 20], [1; 20], [2; 20]];
    assert_eq!(c.validate(), Err(ConfigError::Zero("a roster address")));
}

#[test]
fn every_field_moves_the_config_hash() {
    let base = sample_config().config_hash();
    type Edit = Box<dyn Fn(&mut RoundConfig)>;
    let edits: Vec<Edit> = vec![
        Box::new(|c| c.chain_id += 1),
        Box::new(|c| c.ledger[0] ^= 1),
        Box::new(|c| c.cluster_id[0] ^= 1),
        Box::new(|c| c.base_model_sha256[0] ^= 1),
        Box::new(|c| c.start_adapter_sha256[0] ^= 1),
        Box::new(|c| c.roster.push([0x04; 20])),
        Box::new(|c| c.min_participants += 1),
        Box::new(|c| c.chunk_dim += 1),
        Box::new(|c| c.value_scale_log2 += 1),
        Box::new(|c| c.threshold_pos += 1),
        Box::new(|c| c.threshold_neg -= 1),
        Box::new(|c| c.max_values += 1),
    ];
    for (i, e) in edits.iter().enumerate() {
        let mut c = sample_config();
        e(&mut c);
        assert_ne!(c.config_hash(), base, "edit {i} did not move the hash");
    }
}

#[test]
fn round_ids_are_pinned_and_distinct_per_ordinal() {
    let c = sample_config();
    // keccak256("citrate-fl-round-key/1" ‖ be64(1337) ‖ ledger ‖ cluster ‖ be64(0)),
    // the same packing as FederatedRoundLedger.roundIdOf.
    let mut pre = Vec::new();
    pre.extend_from_slice(b"citrate-fl-round-key/1");
    pre.extend_from_slice(&1337u64.to_be_bytes());
    pre.extend_from_slice(&[0x11; 20]);
    pre.extend_from_slice(&[0x22; 32]);
    pre.extend_from_slice(&0u64.to_be_bytes());
    assert_eq!(c.round_id(0), keccak(&[&pre]));
    assert_ne!(c.round_id(0), c.round_id(1));
}

#[test]
fn a_payload_round_trips_and_is_checked() {
    let p = LoraDeltaPayload::new(sample_config(), 7);
    p.validate().expect("valid");
    let json = serde_json::to_string(&p).expect("json");
    let back: LoraDeltaPayload = serde_json::from_str(&json).expect("parse");
    assert_eq!(back, p);

    let mut wrong = p.clone();
    wrong.ordinal = 8;
    assert_eq!(wrong.validate(), Err(PayloadError::RoundId));

    let mut wrong = p.clone();
    wrong.task = "train".into();
    assert!(matches!(wrong.validate(), Err(PayloadError::Task(_))));

    // An unknown field is a refusal, not something to ignore.
    let mut v: serde_json::Value = serde_json::from_str(&json).expect("value");
    v["epochs"] = serde_json::json!(3);
    assert!(serde_json::from_value::<LoraDeltaPayload>(v).is_err());
}

#[test]
fn the_delta_digest_binds_every_field() {
    let base = delta_digest(&[1; 32], &[2; 20], &[3; 32], &[4; 32], 5, 6);
    assert_ne!(base, delta_digest(&[9; 32], &[2; 20], &[3; 32], &[4; 32], 5, 6));
    assert_ne!(base, delta_digest(&[1; 32], &[9; 20], &[3; 32], &[4; 32], 5, 6));
    assert_ne!(base, delta_digest(&[1; 32], &[2; 20], &[9; 32], &[4; 32], 5, 6));
    assert_ne!(base, delta_digest(&[1; 32], &[2; 20], &[3; 32], &[9; 32], 5, 6));
    assert_ne!(base, delta_digest(&[1; 32], &[2; 20], &[3; 32], &[4; 32], 9, 6));
    assert_ne!(base, delta_digest(&[1; 32], &[2; 20], &[3; 32], &[4; 32], 5, 9));
}

/// `FederatedRoundLedger.commitRound` refuses more participants than 0x0110 accepts (a larger
/// round could never be recomputed by a challenge), so a config that allows one is refused here
/// before any device trains for it. Narrow chunks keep the cell bound from catching it.
#[test]
fn a_roster_larger_than_the_precompile_accepts_is_refused() {
    let mut c = sample_config();
    c.chunk_dim = 1;
    c.roster = (1..=1025u32)
        .map(|i| {
            let mut a = [0u8; 20];
            a[16..].copy_from_slice(&i.to_be_bytes());
            a
        })
        .collect();
    assert_eq!(c.validate(), Err(ConfigError::RosterTooLarge(1025)));
    c.roster.pop();
    c.validate().expect("1024 devices fit 0x0110");
}
