use super::*;
use crate::coordinator_protocol::{Capability, JobSpec};
use crate::fl::belnap::{ConfidenceRule, WeightRule};
use crate::fl::gguf::{encode_f32, OutTensor, Value};
use crate::fl::round::RoundConfig;
use crate::fl::trainer::TrainError;

/// A throwaway key built at run time (no key material in the source).
pub(crate) fn test_wallet(n: u64) -> Wallet {
    Wallet::from_hex(&format!("{:064x}", 0x1000 + n)).expect("wallet")
}

fn kv() -> Vec<(String, Value)> {
    vec![
        ("general.architecture".into(), Value::Str("gemma4".into())),
        ("general.type".into(), Value::Str("adapter".into())),
        ("adapter.type".into(), Value::Str("lora".into())),
        ("adapter.lora.alpha".into(), Value::F32(16.0)),
    ]
}

fn adapter_bytes(a: &[f32], b: &[f32]) -> Vec<u8> {
    encode_f32(
        &kv(),
        &[
            OutTensor {
                name: "blk.0.attn_v.weight.lora_a",
                dims: &[4, 2],
                values: a,
            },
            OutTensor {
                name: "blk.0.attn_v.weight.lora_b",
                dims: &[2, 4],
                values: b,
            },
        ],
    )
    .expect("encode")
}

/// Test double: "trains" by adding a fixed step to every lora_b value, or
/// writes a differently shaped adapter when asked to.
struct StepTrainer {
    step: f32,
    wrong_shape: bool,
}

#[async_trait::async_trait]
impl LoraTrainer for StepTrainer {
    fn id(&self) -> String {
        "step-trainer-test".into()
    }
    async fn train(&self, req: &TrainRequest) -> Result<(), TrainError> {
        let a = [0.01f32; 8];
        let bytes = if self.wrong_shape {
            encode_f32(
                &kv(),
                &[OutTensor {
                    name: "blk.0.attn_v.weight.lora_a",
                    dims: &[8],
                    values: &a,
                }],
            )
            .expect("encode")
        } else {
            adapter_bytes(&a, &[self.step; 8])
        };
        std::fs::write(&req.out, bytes)?;
        Ok(())
    }
}

struct Fixture {
    dir: PathBuf,
    cfg: RoundConfig,
    device: DeviceConfig,
    wallet: Wallet,
}

const DATASET: &str = r#"{"messages":[{"role":"user","content":"mint"},{"role":"assistant","content":"done"}],"metadata":{"model":"m","workflow":"hello-mint","step":"deploy","verifiers":["forge-test"]}}
{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}],"metadata":{"model":"m","workflow":null,"step":null,"verifiers":["exit-code"]}}
"#;

fn fixture(tag: &str) -> Fixture {
    let dir = std::env::temp_dir().join(format!("fl-runner-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let wallet = test_wallet(1);
    let me = wallet.address().to_fixed_bytes();
    let base = b"not really a model, but hash-verified".to_vec();
    let start = adapter_bytes(&[0.01; 8], &[0.0; 8]);
    let mut roster = vec![me, [0x01; 20], [0xfe; 20]];
    roster.sort();
    let cfg = RoundConfig {
        chain_id: 1337,
        ledger: [0x11; 20],
        cluster_id: [0x22; 32],
        base_model_sha256: sha256(&base),
        start_adapter_sha256: sha256(&start),
        roster,
        min_participants: 3,
        chunk_dim: 4,
        value_scale_log2: 8,
        threshold_pos: 32768,
        threshold_neg: -32768,
        confidence: ConfidenceRule::Nonzero,
        weight: WeightRule::Uniform,
        max_values: 1 << 20,
    };
    let device = DeviceConfig {
        store: dir.join("store"),
        dataset: dir.join("export.jsonl"),
        consent_file: Some(dir.join("consent.json")),
    };
    std::fs::create_dir_all(dir.join("store/models")).expect("models");
    std::fs::create_dir_all(dir.join("store/adapters")).expect("adapters");
    std::fs::write(device.model_path(&cfg.base_model_sha256), &base).expect("base");
    std::fs::write(device.adapter_path(&cfg.start_adapter_sha256), &start).expect("start");
    std::fs::write(&device.dataset, DATASET).expect("dataset");
    let round = cfg.round_id(0);
    std::fs::write(
        dir.join("consent.json"),
        format!("{{\"rounds\":[\"{}\"]}}", hex0x(&round)),
    )
    .expect("consent");
    Fixture {
        dir,
        cfg,
        device,
        wallet,
    }
}

fn job(cfg: &RoundConfig) -> JobSpec {
    JobSpec::new(
        "fl-0-0",
        Capability::Federated,
        serde_json::to_value(LoraDeltaPayload::new(cfg.clone(), 0)).expect("payload"),
    )
}

fn runner(f: &Fixture, trainer: Option<Box<dyn LoraTrainer>>) -> LoraDeltaRunner {
    LoraDeltaRunner::new(f.device.clone(), trainer, f.wallet.clone())
}

fn step(step: f32) -> Option<Box<dyn LoraTrainer>> {
    Some(Box::new(StepTrainer {
        step,
        wrong_shape: false,
    }))
}

#[tokio::test]
async fn a_round_produces_a_signed_verifiable_delta() {
    let f = fixture("ok");
    let out = runner(&f, step(0.5)).run(&job(&f.cfg)).await.expect("run");
    let r = &out.result;
    // The signature recovers to this device.
    let who = Wallet::recover_address(&r.digest(), &hex::decode(&r.signature[2..]).expect("hex"))
        .expect("recover");
    assert_eq!(who, f.wallet.address());
    // The artifact is what the result commits to.
    assert_eq!(sha256(&out.artifact), r.delta_sha256);
    let art = Artifact::decode(&out.artifact, f.cfg.max_values).expect("decode");
    assert_eq!(art.round_id, f.cfg.round_id(0));
    assert_eq!(art.worker, f.wallet.address().to_fixed_bytes());
    assert_eq!(delta::delta_root(&art.values, 4).expect("root"), r.delta_root);
    // lora_a did not move; every lora_b moved by 0.5 * 2^8 on the Q16 grid.
    assert_eq!(&art.values[..8], &[0; 8]);
    assert_eq!(&art.values[8..], &[128 * 65536; 8]);
    assert_eq!(r.examples, 2);
    assert_eq!(r.trainer, "step-trainer-test");
    // The submission payload parses back to the same result.
    let back: LoraDeltaResult = serde_json::from_str(&out.result_json).expect("json");
    assert_eq!(&back, r);
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn a_device_off_the_roster_declines() {
    let mut f = fixture("roster");
    f.wallet = test_wallet(2);
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::NotInRoster(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn without_consent_for_this_round_the_device_declines() {
    let f = fixture("consent");
    std::fs::write(f.dir.join("consent.json"), r#"{"rounds":[]}"#).expect("consent");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::NoConsent(_))
    ));
    std::fs::remove_file(f.dir.join("consent.json")).expect("rm");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::NoConsent(_))
    ));
    let mut no_file = runner(&f, step(0.5));
    no_file.device.consent_file = None;
    assert!(matches!(
        no_file.run(&job(&f.cfg)).await,
        Err(LoraRunError::NoConsent(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn a_second_job_of_the_same_round_is_left_for_another_device() {
    let f = fixture("twice");
    let r = runner(&f, step(0.5));
    r.run(&job(&f.cfg)).await.expect("first");
    r.mark_contributed(&f.cfg.round_id(0)).expect("mark");
    assert!(matches!(
        r.run(&job(&f.cfg)).await,
        Err(LoraRunError::AlreadyContributed(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn no_trainer_means_an_honest_decline() {
    let f = fixture("notrainer");
    assert!(matches!(
        runner(&f, None).run(&job(&f.cfg)).await,
        Err(LoraRunError::NoTrainer)
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn unstaged_or_wrong_inputs_are_refused_before_training() {
    let f = fixture("inputs");
    std::fs::write(f.device.model_path(&f.cfg.base_model_sha256), b"swapped").expect("swap");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::HashMismatch { .. })
    ));
    std::fs::remove_file(f.device.model_path(&f.cfg.base_model_sha256)).expect("rm");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::NotStaged { .. })
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn only_verified_trajectories_are_trained_on() {
    let f = fixture("data");
    std::fs::write(
        &f.device.dataset,
        r#"{"messages":[{"role":"user","content":"x"}],"metadata":{"model":"m","workflow":null,"step":null,"verifiers":[]}}"#,
    )
    .expect("write");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::Dataset(_))
    ));
    std::fs::write(&f.device.dataset, "free text, not an export\n").expect("write");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::Dataset(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn a_trained_adapter_of_another_shape_is_refused() {
    let f = fixture("shape");
    let t: Option<Box<dyn LoraTrainer>> = Some(Box::new(StepTrainer {
        step: 0.5,
        wrong_shape: true,
    }));
    assert!(matches!(
        runner(&f, t).run(&job(&f.cfg)).await,
        Err(LoraRunError::Adapter(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn a_tampered_payload_is_refused() {
    let f = fixture("payload");
    let mut j = job(&f.cfg);
    j.payload["ordinal"] = serde_json::json!(1);
    assert!(matches!(
        runner(&f, step(0.5)).run(&j).await,
        Err(LoraRunError::BadPayload(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[test]
fn the_export_check_reads_the_s9_3_shape() {
    assert_eq!(verify_dataset(DATASET.as_bytes()), Ok(2));
    let system = r#"{"messages":[{"role":"system","content":"s"}],"metadata":{"model":"m","workflow":null,"step":null,"verifiers":["v"]}}"#;
    assert!(verify_dataset(system.as_bytes()).is_err());
    assert!(verify_dataset(b"").is_err());
    assert!(verify_dataset(&[0xff, 0xfe]).is_err());
}

#[tokio::test]
async fn an_unreadable_consent_file_is_an_error_not_a_silent_no() {
    let f = fixture("consent-io");
    // A directory where the consent file should be: reading it fails with something other
    // than NotFound, which must surface instead of reading as "no consent".
    let _ = std::fs::remove_file(f.dir.join("consent.json"));
    std::fs::create_dir_all(f.dir.join("consent.json")).expect("dir");
    assert!(matches!(
        runner(&f, step(0.5)).run(&job(&f.cfg)).await,
        Err(LoraRunError::Io(_))
    ));
    let _ = std::fs::remove_dir_all(&f.dir);
}
