use super::*;
use crate::job::{Capability, JobSpec};
use citrate_training_worker::fl::belnap::{ConfidenceRule, WeightRule};
use citrate_training_worker::fl::gguf::{encode_f32, OutTensor, Value};
use citrate_training_worker::fl::round::TASK_LORA_DELTA;

/// TEST ORACLE ONLY: a transcription of the `0x0110` kernel
/// (`core/execution/src/precompiles/q16/belnap.rs`, `StandardBelnap`), used so
/// these tests can aggregate arbitrary deltas without a node. It is pinned to
/// the real precompile by `the_test_oracle_matches_the_live_precompile_vector`
/// (the vector was produced by `0x0110` on a local Citrate devnet and is pinned
/// on the chain side by the kernel itself). Production code never computes an
/// aggregate.
struct OracleBelnap;

fn sat_mul(a: i64, b: i64) -> i64 {
    let p = (i128::from(a) * i128::from(b)) >> 16;
    p.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[async_trait::async_trait]
impl BelnapBackend for OracleBelnap {
    async fn aggregate(&self, input: &[u8]) -> Result<Vec<u8>, RoundError> {
        let rd = |o: usize| i64::from_be_bytes(input[o..o + 8].try_into().expect("8"));
        let dim = u32::from_be_bytes(input[0..4].try_into().expect("4")) as usize;
        let n = u32::from_be_bytes(input[4..8].try_into().expect("4")) as usize;
        let emb = 8;
        let conf = emb + 8 * n * dim;
        let w = conf + 8 * n * dim;
        let tpos = rd(w + 8 * n);
        let mut vals = Vec::new();
        let mut states = Vec::new();
        for d in 0..dim {
            let mut acc: i64 = 0;
            for i in 0..n {
                acc = acc.saturating_add(sat_mul(rd(w + 8 * i), rd(emb + 8 * (i * dim + d))));
            }
            vals.extend_from_slice(&acc.to_be_bytes());
            let (mut agree, mut oppose) = (false, false);
            for i in 0..n {
                if rd(w + 8 * i) <= 0 {
                    continue;
                }
                let e = rd(emb + 8 * (i * dim + d));
                let c = rd(conf + 8 * (i * dim + d));
                if c >= tpos {
                    if e >= 0 {
                        agree = true
                    } else {
                        oppose = true
                    }
                }
            }
            states.push(match (agree, oppose) {
                (false, false) => 0u8,
                (true, true) => 3,
                _ => 1,
            });
        }
        vals.extend_from_slice(&states);
        Ok(vals)
    }
}

/// A backend standing in for an unreachable node.
struct DownBelnap;

#[async_trait::async_trait]
impl BelnapBackend for DownBelnap {
    async fn aggregate(&self, _input: &[u8]) -> Result<Vec<u8>, RoundError> {
        Err(RoundError::Node("connection refused".into()))
    }
}

/// The vector `0x0110` returned on a local devnet for the golden chunk
/// (training-worker `fl::belnap` tests carry the same pair).
const GOLDEN_INPUT_ROWS: [[i64; 4]; 3] = [
    [65536, -32768, 0, 100],
    [32768, -32768, 0, 50],
    [-16384, 16384, 0, 25],
];
const GOLDEN_OUTPUT_HEX: &str =
    "0000000000006aa9ffffffffffffbfff0000000000000000000000000000003903030001";

fn rules() -> citrate_training_worker::fl::belnap::ChunkRules {
    citrate_training_worker::fl::belnap::ChunkRules {
        confidence: ConfidenceRule::Nonzero,
        weight: WeightRule::Uniform,
        threshold_pos: 32768,
        threshold_neg: -32768,
    }
}

#[tokio::test]
async fn the_test_oracle_matches_the_live_precompile_vector() {
    let refs: Vec<&[i64]> = GOLDEN_INPUT_ROWS.iter().map(|r| r.as_slice()).collect();
    let input = chunk_input(&refs, &rules()).expect("input");
    let out = OracleBelnap.aggregate(&input).await.expect("oracle");
    assert_eq!(hex::encode(out), GOLDEN_OUTPUT_HEX);
}

#[test]
fn bytes_calls_and_returns_follow_the_abi() {
    let data = vec![0xabu8; 33];
    let call = encode_bytes_call("belnapAggregate(bytes)", &data);
    assert_eq!(&call[..4], &selector("belnapAggregate(bytes)"));
    assert_eq!(call.len(), 4 + 32 + 32 + 64);
    assert_eq!(call[4 + 31], 0x20);
    assert_eq!(call[4 + 32 + 31], 33);
    // A return is the same shape without the selector.
    let back = decode_bytes_return(&call[4..]).expect("decode");
    assert_eq!(back, data);
    assert!(decode_bytes_return(&[0u8; 10]).is_err());
    let mut lying = call[4..].to_vec();
    lying[32 + 31] = 200;
    assert!(decode_bytes_return(&lying).is_err());
}

fn kv() -> Vec<(String, Value)> {
    vec![
        ("general.architecture".into(), Value::Str("gemma4".into())),
        ("general.type".into(), Value::Str("adapter".into())),
        ("adapter.type".into(), Value::Str("lora".into())),
        ("adapter.lora.alpha".into(), Value::F32(16.0)),
    ]
}

fn adapter(dir: &Path, name: &str, a: &[f32], b: &[f32]) -> (std::path::PathBuf, Adapter) {
    let p = dir.join(name);
    let bytes = encode_f32(
        &kv(),
        &[
            OutTensor {
                name: "blk.0.attn_q.weight.lora_a",
                dims: &[5, 2],
                values: a,
            },
            OutTensor {
                name: "blk.0.attn_q.weight.lora_b",
                dims: &[2, 3],
                values: b,
            },
        ],
    )
    .expect("encode");
    std::fs::write(&p, bytes).expect("write");
    let ad = Adapter::load(&p).expect("load");
    (p, ad)
}

fn wallet(n: u64) -> Wallet {
    Wallet::from_hex(&format!("{:064x}", 0x3000 + n)).expect("wallet")
}

struct World {
    dir: std::path::PathBuf,
    cfg: RoundConfig,
    start: Adapter,
    state: State,
    deltas: std::path::PathBuf,
    wallets: Vec<Wallet>,
}

const A0: [f32; 10] = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5];

/// Four roster devices; `steps[i]` is how far device i moved every lora_b
/// value. Devices with `None` submit nothing.
fn world(tag: &str, steps: &[Option<f32>]) -> World {
    let dir = std::env::temp_dir().join(format!("fl-round-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let deltas = dir.join("deltas");
    std::fs::create_dir_all(&deltas).expect("deltas");
    let (start_path, start) = adapter(&dir, "start.gguf", &A0, &[0.0; 6]);
    let wallets: Vec<Wallet> = (0..steps.len() as u64).map(wallet).collect();
    let mut roster: Vec<Addr> = wallets.iter().map(|w| w.address().to_fixed_bytes()).collect();
    roster.sort();
    let cfg = RoundConfig {
        chain_id: 1337,
        ledger: [0x11; 20],
        cluster_id: [0x22; 32],
        base_model_sha256: [0x33; 32],
        start_adapter_sha256: citrate_training_worker::fl::sha256_file(&start_path).expect("sha"),
        roster,
        min_participants: 3,
        chunk_dim: 4,
        value_scale_log2: 8,
        threshold_pos: 32768,
        threshold_neg: -32768,
        confidence: ConfidenceRule::Nonzero,
        weight: WeightRule::Uniform,
        max_values: 1 << 16,
    };
    let mut state = State::default();
    for (i, (w, step)) in wallets.iter().zip(steps).enumerate() {
        let id = JobId(format!("fl-{i}"));
        state.add_job(JobSpec::new(
            id.0.clone(),
            Capability::Federated,
            serde_json::to_value(LoraDeltaPayload::new(cfg.clone(), 0)).expect("payload"),
        ));
        let Some(step) = step else { continue };
        let (_, trained) = adapter(&dir, &format!("t{i}.gguf"), &A0, &[*step; 6]);
        let me = w.address().to_fixed_bytes();
        let art = Artifact {
            value_scale_log2: 8,
            round_id: cfg.round_id(0),
            worker: me,
            start_adapter_sha256: cfg.start_adapter_sha256,
            trained_adapter_sha256: [7; 32],
            manifest_hash: delta::manifest_hash(&start),
            chunk_dim: 4,
            values: delta::compute(&start, &trained, 8).expect("delta"),
        };
        let bytes = art.encode();
        let sha = sha256(&bytes);
        std::fs::write(deltas.join(format!("{}.fld", hex::encode(sha))), &bytes).expect("art");
        let mut r = LoraDeltaResult {
            task: TASK_LORA_DELTA.into(),
            round_id: cfg.round_id(0),
            worker: me,
            delta_root: delta::delta_root(&art.values, 4).expect("root"),
            delta_sha256: sha,
            n_values: art.values.len() as u64,
            chunk_dim: 4,
            start_adapter_sha256: cfg.start_adapter_sha256,
            trained_adapter_sha256: [7; 32],
            dataset_sha256: [6; 32],
            examples: 3,
            trainer: "test".into(),
            seconds: 1.0,
            signature: String::new(),
        };
        r.signature = hex0x(&w.sign_digest_recoverable(&r.digest()).expect("sign"));
        let rec = state.jobs.get_mut(&id).expect("job");
        rec.status = JobStatus::Done {
            worker: w.address(),
            at: 1,
        };
        rec.result = Some(serde_json::to_string(&r).expect("json"));
    }
    World {
        dir,
        cfg,
        start,
        state,
        deltas,
        wallets,
    }
}

#[tokio::test]
async fn three_devices_aggregate_into_their_mean_and_a_consistent_bundle() {
    let w = world("ok", &[Some(0.25), Some(0.5), Some(0.75), None]);
    let (contribs, excluded) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    assert!(excluded.is_empty(), "{excluded:?}");
    assert_eq!(contribs.len(), 3);
    // Canonical order: ascending worker address.
    assert!(contribs.windows(2).all(|p| p[0].result.worker < p[1].result.worker));
    let out = aggregate(&w.cfg, 0, contribs, excluded, &w.start, &OracleBelnap)
        .await
        .expect("aggregate");
    let b = &out.bundle;
    assert_eq!(b.n_values, 16);
    assert_eq!(b.chunks, 4);
    assert_eq!(b.participants.len(), 3);
    check_bundle_roots(b).expect("roots follow from leaves");
    assert_eq!(b.adapter_sha256, sha256(&out.merged_adapter));
    // lora_a unmoved (Neither), every lora_b moved the same way (True).
    assert_eq!(b.state_counts, [10, 6, 0, 0]);
    // The merged adapter is start + the (floor-weighted) mean of the deltas.
    let p = w.dir.join("merged.gguf");
    std::fs::write(&p, &out.merged_adapter).expect("write");
    let m = Adapter::load(&p).expect("merged adapter loads");
    assert_eq!(m.values[0], A0.to_vec());
    for v in &m.values[1] {
        assert!((v - 0.5).abs() < 1.0e-4, "{v}");
    }
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[tokio::test]
async fn fewer_than_the_minimum_is_no_round() {
    let w = world("few", &[Some(0.25), Some(0.5), None, None]);
    let (contribs, excluded) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    assert_eq!(contribs.len(), 2);
    assert!(matches!(
        aggregate(&w.cfg, 0, contribs, excluded, &w.start, &OracleBelnap).await,
        Err(RoundError::TooFew { got: 2, need: 3 })
    ));
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[tokio::test]
async fn a_node_that_is_down_fails_the_round_rather_than_guessing() {
    let w = world("down", &[Some(0.25), Some(0.5), Some(0.75)]);
    let (c, e) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    assert!(matches!(
        aggregate(&w.cfg, 0, c, e, &w.start, &DownBelnap).await,
        Err(RoundError::Node(_))
    ));
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[test]
fn forged_tampered_and_foreign_contributions_are_excluded_with_reasons() {
    let mut w = world("excl", &[Some(0.25), Some(0.5), Some(0.75), Some(1.0)]);
    let ids: Vec<JobId> = (0..4).map(|i| JobId(format!("fl-{i}"))).collect();
    // fl-0: the signature is replaced with one from another key.
    {
        let rec = w.state.jobs.get_mut(&ids[0]).expect("job");
        let mut r: LoraDeltaResult =
            serde_json::from_str(rec.result.as_deref().expect("result")).expect("parse");
        r.signature = hex0x(&wallet(99).sign_digest_recoverable(&r.digest()).expect("sign"));
        rec.result = Some(serde_json::to_string(&r).expect("json"));
    }
    // fl-1: the coordinator recorded a different submitter than the result names.
    w.state.jobs.get_mut(&ids[1]).expect("job").status = JobStatus::Done {
        worker: wallet(98).address(),
        at: 1,
    };
    // fl-2: the stored artifact is altered after signing.
    {
        let r: LoraDeltaResult = serde_json::from_str(
            w.state.jobs[&ids[2]].result.as_deref().expect("result"),
        )
        .expect("parse");
        let p = w.deltas.join(format!("{}.fld", hex::encode(r.delta_sha256)));
        let mut bytes = std::fs::read(&p).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&p, bytes).expect("write");
    }
    let (c, e) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    assert_eq!(c.len(), 1);
    assert_eq!(e.len(), 3, "{e:?}");
    let reasons: Vec<&str> = e.iter().map(|x| x.reason.as_str()).collect();
    assert!(reasons.iter().any(|r| r.contains("does not recover")));
    assert!(reasons.iter().any(|r| r.contains("other than its submitter")));
    assert!(reasons.iter().any(|r| r.contains("does not hash")));
    let _ = std::fs::remove_dir_all(&w.dir);
    let _ = &w.wallets;
}

#[tokio::test]
async fn chunk_proofs_verify_against_the_committed_roots() {
    let w = world("proof", &[Some(0.25), Some(-0.5), Some(0.75)]);
    let (c, e) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    let out = aggregate(&w.cfg, 0, c, e, &w.start, &OracleBelnap)
        .await
        .expect("aggregate");
    let b = &out.bundle;
    for chunk in 0..b.chunks {
        let p = chunk_proof(b, &w.deltas, chunk).expect("proof");
        let input = hex::decode(&p.input[2..]).expect("hex");
        let ip = parse_list(&p.input_proof).expect("ip");
        assert!(tree::verify(&b.input_root, chunk, &keccak(&[&input]), &ip));
        let op = parse_list(&p.output_proof).expect("op");
        let oh = citrate_training_worker::fl::parse_hex::<32>(&p.output_hash).expect("oh");
        assert!(tree::verify(&b.output_root, chunk, &oh, &op));
        // Row k of the input is participant k's committed row.
        let dim = u32::from_be_bytes(input[0..4].try_into().expect("4")) as usize;
        for r in &p.rows {
            let k = usize::from(r.participant);
            let row = &input[8 + k * dim * 8..8 + (k + 1) * dim * 8];
            let rh = citrate_training_worker::fl::parse_hex::<32>(&r.row_hash).expect("rh");
            assert_eq!(keccak(&[row]), rh);
            let dr = citrate_training_worker::fl::parse_hex::<32>(&r.delta_root).expect("dr");
            assert!(tree::verify(&dr, chunk, &rh, &parse_list(&r.row_proof).expect("rp")));
            let worker = citrate_training_worker::fl::parse_hex::<20>(&r.worker).expect("w");
            assert!(tree::verify(
                &b.participants_root,
                u32::from(r.participant),
                &participant_payload(&worker, &dr),
                &parse_list(&r.participant_proof).expect("pp"),
            ));
        }
    }
    // Opposite signs on lora_b make those coordinates Both.
    assert_eq!(b.state_counts, [10, 0, 0, 6]);
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[tokio::test]
async fn a_bundle_whose_roots_do_not_follow_is_detected() {
    let w = world("tamper", &[Some(0.25), Some(0.5), Some(0.75)]);
    let (c, e) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    let out = aggregate(&w.cfg, 0, c, e, &w.start, &OracleBelnap)
        .await
        .expect("aggregate");
    let mut b = out.bundle.clone();
    b.output_hashes[1] = hex0x(&[0xee; 32]);
    assert!(check_bundle_roots(&b).is_err());
    let mut b = out.bundle.clone();
    b.adapter_sha256[0] ^= 1;
    assert!(check_bundle_roots(&b).is_err());
    // A proof is never produced for an input the bundle did not commit.
    let mut b = out.bundle;
    b.input_hashes[0] = hex0x(&[0xee; 32]);
    assert!(chunk_proof(&b, &w.deltas, 0).is_err());
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[tokio::test]
async fn the_commit_intent_is_unsigned_calldata_for_the_ledger() {
    let w = world("intent", &[Some(0.25), Some(0.5), Some(0.75)]);
    let (c, e) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    let out = aggregate(&w.cfg, 0, c, e, &w.start, &OracleBelnap)
        .await
        .expect("aggregate");
    let i = commit_intent(&out.bundle);
    assert_eq!(i.to, hex0x(&w.cfg.ledger));
    assert_eq!(i.chain_id, 1337);
    let data = hex::decode(&i.data[2..]).expect("hex");
    assert_eq!(data.len(), 4 + 10 * 32);
    assert_eq!(&data[..4], &selector(COMMIT_SIG));
    assert_eq!(&data[4..36], &w.cfg.cluster_id);
    assert_eq!(&data[4 + 9 * 32..], &out.bundle.adapter_sha256);
    // No signature anywhere: the intent is for the operator's ceremony.
    let json = serde_json::to_value(&i).expect("json");
    assert!(json.get("signature").is_none());
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[test]
fn the_record_digest_matches_abi_encode() {
    // Word-by-word abi.encode of (uint256, address, bytes32 x6, uint256 x3).
    let d = record_digest(1337, &[0x11; 20], &[1; 32], &[2; 32], &[3; 32], &[4; 32], &[5; 32], &[6; 32], 7, 8, 9);
    let mut pre = Vec::new();
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&1337u64.to_be_bytes());
    pre.extend_from_slice(&w);
    let mut a = [0u8; 32];
    a[12..].copy_from_slice(&[0x11; 20]);
    pre.extend_from_slice(&a);
    for b in 1..=6u8 {
        pre.extend_from_slice(&[b; 32]);
    }
    for v in [7u64, 8, 9] {
        let mut w = [0u8; 32];
        w[24..].copy_from_slice(&v.to_be_bytes());
        pre.extend_from_slice(&w);
    }
    assert_eq!(d, keccak(&[&pre]));
}

#[tokio::test]
async fn recomputed_roots_match_an_honest_bundle_and_expose_an_edited_one() {
    let w = world("reroot", &[Some(0.25), Some(0.5), Some(0.75)]);
    let (c, e) = collect(&w.state, &w.cfg, 0, &w.deltas, &w.start);
    let out = aggregate(&w.cfg, 0, c, e, &w.start, &OracleBelnap)
        .await
        .expect("aggregate");
    let b = out.bundle;
    let r = recompute_roots(&b).expect("roots");
    assert_eq!(r.participants_root, b.participants_root);
    assert_eq!(r.input_root, b.input_root);
    assert_eq!(r.output_root, b.output_root);
    assert_eq!(r.record_digest, b.record_digest);
    let mut edited = b.clone();
    edited.output_hashes[2] = hex0x(&[0x01; 32]);
    let r2 = recompute_roots(&edited).expect("roots");
    assert_ne!(r2.output_root, b.output_root);
    assert_eq!(r2.input_root, b.input_root);
    assert_ne!(r2.record_digest, b.record_digest);
    let _ = std::fs::remove_dir_all(&w.dir);
}

#[test]
fn the_node_url_follows_the_outbound_policy() {
    assert!(RpcBelnap::new("http://127.0.0.1:8545", [1; 20]).is_ok());
    assert!(RpcBelnap::new("https://rpc.example", [1; 20]).is_ok());
    assert!(RpcBelnap::new("http://203.0.113.7:8545", [1; 20]).is_err());
    assert!(RpcBelnap::new("ftp://x", [1; 20]).is_err());
}
