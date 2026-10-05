//! `citrate-fl-round`: the coordinator operator's tool for federated LoRA rounds
//! (HUP-S9.2). It reads and writes files and calls a Citrate node's `eth_call`;
//! it never signs and never sends a transaction. The commit it prepares is an
//! unsigned intent for the operator's ceremony.
//!
//! ```text
//! citrate-fl-round config-hash  --config round.json [--ordinal N]
//! citrate-fl-round jobs         --config round.json --ordinal N [--lease-secs S]
//! citrate-fl-round init-adapter --base base.gguf --rank R --alpha A
//!                               --targets attn_q,attn_v --seed S --out start.gguf
//! citrate-fl-round aggregate    --config round.json --ordinal N --state state.json
//!                               --deltas DIR --start start.gguf --rpc URL --out DIR
//! citrate-fl-round proof        --bundle bundle.json --deltas DIR --chunk C
//! citrate-fl-round check-bundle --bundle bundle.json
//! citrate-fl-round roots        --bundle bundle.json
//! citrate-fl-round intent       --bundle bundle.json
//! ```
//!
//! `roots` recomputes a bundle's roots and record digest from its leaves (it
//! does not trust the ones the bundle states); `intent` prints the unsigned
//! `commitRound` intent for a bundle whose roots follow from its leaves.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context};
use citrate_training_coordinator::fl_round::{
    aggregate, check_bundle_roots, chunk_proof, collect, commit_intent, recompute_roots, Bundle,
    RpcBelnap,
};
use citrate_training_coordinator::job::{Capability, JobSpec};
use citrate_training_coordinator::store::Store;
use citrate_training_worker::fl::gguf::{self, Adapter, OutTensor, Value};
use citrate_training_worker::fl::round::{LoraDeltaPayload, RoundConfig};
use citrate_training_worker::fl::{hex0x, sha256};

fn args() -> anyhow::Result<(String, BTreeMap<String, String>)> {
    let mut it = std::env::args().skip(1);
    let cmd = it
        .next()
        .ok_or_else(|| anyhow!("missing subcommand (see --help in the source)"))?;
    let mut map = BTreeMap::new();
    while let Some(k) = it.next() {
        let key = k
            .strip_prefix("--")
            .ok_or_else(|| anyhow!("expected --flag, got {k:?}"))?
            .to_string();
        let v = it.next().ok_or_else(|| anyhow!("--{key} needs a value"))?;
        if map.insert(key.clone(), v).is_some() {
            bail!("--{key} given twice");
        }
    }
    Ok((cmd, map))
}

fn need<'a>(m: &'a BTreeMap<String, String>, k: &str) -> anyhow::Result<&'a str> {
    m.get(k)
        .map(String::as_str)
        .ok_or_else(|| anyhow!("--{k} is required"))
}

fn num<T: std::str::FromStr>(m: &BTreeMap<String, String>, k: &str) -> anyhow::Result<T>
where
    T::Err: std::fmt::Display,
{
    need(m, k)?.parse().map_err(|e| anyhow!("--{k}: {e}"))
}

fn load_config(path: &str) -> anyhow::Result<RoundConfig> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let cfg: RoundConfig = serde_json::from_str(&raw).with_context(|| format!("parse {path}"))?;
    cfg.validate().map_err(|e| anyhow!("{path}: {e}"))?;
    Ok(cfg)
}

fn print(v: &impl serde::Serialize) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

/// splitmix64: a fixed, portable generator so an initial adapter is
/// reproducible from its seed on any machine.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-bound, bound).
    fn uniform(&mut self, bound: f32) -> f32 {
        let x = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (2.0 * x - 1.0) * bound
    }
}

/// A fresh LoRA adapter for `base`: `lora_a` drawn uniformly in
/// ±1/sqrt(fan_in) (the usual Kaiming-uniform bound), `lora_b` zero, so the
/// adapter starts as an exact no-op on the base model.
fn init_adapter(m: &BTreeMap<String, String>) -> anyhow::Result<()> {
    let base = PathBuf::from(need(m, "base")?);
    let rank: u64 = num(m, "rank")?;
    let alpha: f32 = num(m, "alpha")?;
    let seed: u64 = num(m, "seed")?;
    let out = PathBuf::from(need(m, "out")?);
    if rank == 0 || rank > 256 {
        bail!("--rank must be 1..=256");
    }
    let targets: Vec<String> = need(m, "targets")?
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let h = gguf::read_header_path(&base).with_context(|| format!("read {}", base.display()))?;
    let arch = h
        .get("general.architecture")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("base model has no general.architecture"))?
        .to_string();
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for t in &h.tensors {
        let Some(rest) = t.name.strip_prefix("blk.") else {
            continue;
        };
        let mut parts = rest.splitn(2, '.');
        let (Some(layer), Some(tail)) = (parts.next(), parts.next()) else {
            continue;
        };
        if layer.parse::<u32>().is_err() || t.dims.len() != 2 {
            continue;
        }
        let Some(target) = tail.strip_suffix(".weight") else {
            continue;
        };
        if !targets.iter().any(|x| x == target) {
            continue;
        }
        let (ne0, ne1) = (t.dims[0], t.dims[1]);
        let mut rng = SplitMix(
            seed ^ u64::from_be_bytes(
                sha256(t.name.as_bytes())[..8]
                    .try_into()
                    .map_err(|_| anyhow!("hash"))?,
            ),
        );
        let bound = 1.0 / (ne0 as f32).sqrt();
        let a: Vec<f32> = (0..ne0 * rank).map(|_| rng.uniform(bound)).collect();
        tensors.push((format!("{}.lora_a", t.name), vec![ne0, rank], a));
        tensors.push((
            format!("{}.lora_b", t.name),
            vec![rank, ne1],
            vec![0.0; (rank * ne1) as usize],
        ));
    }
    if tensors.is_empty() {
        bail!("no 2-D blk.N.<target>.weight tensors match --targets {targets:?}");
    }
    let kv = vec![
        ("general.architecture".to_string(), Value::Str(arch)),
        ("general.type".to_string(), Value::Str("adapter".into())),
        ("adapter.type".to_string(), Value::Str("lora".into())),
        ("adapter.lora.alpha".to_string(), Value::F32(alpha)),
    ];
    let outs: Vec<OutTensor<'_>> = tensors
        .iter()
        .map(|(n, d, v)| OutTensor {
            name: n,
            dims: d,
            values: v,
        })
        .collect();
    let bytes = gguf::encode_f32(&kv, &outs)?;
    std::fs::write(&out, &bytes)?;
    let check = Adapter::load(&out)?;
    let values: u64 = check.values.iter().map(|v| v.len() as u64).sum();
    print(&serde_json::json!({
        "out": out.display().to_string(),
        "sha256": hex::encode(sha256(&bytes)),
        "tensors": check.header.tensors.len(),
        "values": values,
    }))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let (cmd, m) = args()?;
    match cmd.as_str() {
        "config-hash" => {
            let cfg = load_config(need(&m, "config")?)?;
            let ordinal: u64 = m
                .get("ordinal")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(0);
            print(&serde_json::json!({
                "config_hash": hex0x(&cfg.config_hash()),
                "ordinal": ordinal,
                "round_id": hex0x(&cfg.round_id(ordinal)),
            }))
        }
        "jobs" => {
            let cfg = load_config(need(&m, "config")?)?;
            let ordinal: u64 = num(&m, "ordinal")?;
            let lease: u64 = m
                .get("lease-secs")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(6 * 3600);
            let payload = serde_json::to_value(LoraDeltaPayload::new(cfg.clone(), ordinal))?;
            let tag = hex::encode(&cfg.round_id(ordinal)[..6]);
            let jobs: Vec<JobSpec> = (0..cfg.roster.len())
                .map(|i| {
                    JobSpec::new(
                        format!("fl-{tag}-{i}"),
                        Capability::Federated,
                        payload.clone(),
                    )
                    .with_lease_secs(lease)
                })
                .collect();
            print(&jobs)
        }
        "init-adapter" => init_adapter(&m),
        "aggregate" => {
            let cfg = load_config(need(&m, "config")?)?;
            let ordinal: u64 = num(&m, "ordinal")?;
            let state = Store::new(need(&m, "state")?).load()?;
            let deltas = PathBuf::from(need(&m, "deltas")?);
            let start_path = PathBuf::from(need(&m, "start")?);
            let got = citrate_training_worker::fl::sha256_file(&start_path)?;
            if got != cfg.start_adapter_sha256 {
                bail!("--start does not hash to the round's start_adapter_sha256");
            }
            let start = Adapter::load(&start_path)?;
            let out = PathBuf::from(need(&m, "out")?);
            std::fs::create_dir_all(&out)?;
            let (contribs, excluded) = collect(&state, &cfg, ordinal, &deltas, &start);
            for e in &excluded {
                eprintln!("excluded {}: {}", e.job, e.reason);
            }
            let backend = RpcBelnap::new(need(&m, "rpc")?, cfg.ledger)?;
            let outcome = aggregate(&cfg, ordinal, contribs, excluded, &start, &backend).await?;
            let b = &outcome.bundle;
            let adapter_name = format!("merged-{}.gguf", hex::encode(b.adapter_sha256));
            std::fs::write(out.join(&adapter_name), &outcome.merged_adapter)?;
            std::fs::write(out.join("bundle.json"), serde_json::to_vec_pretty(b)?)?;
            let intent = commit_intent(b);
            std::fs::write(
                out.join("commit-intent.json"),
                serde_json::to_vec_pretty(&intent)?,
            )?;
            print(&serde_json::json!({
                "round_id": hex0x(&b.round_id),
                "participants": b.participants.len(),
                "excluded": b.excluded.len(),
                "chunks": b.chunks,
                "n_values": b.n_values,
                "participants_root": hex0x(&b.participants_root),
                "input_root": hex0x(&b.input_root),
                "output_root": hex0x(&b.output_root),
                "adapter_sha256": hex::encode(b.adapter_sha256),
                "record_digest": hex0x(&b.record_digest),
                "state_counts": b.state_counts,
                "merged_adapter": out.join(adapter_name).display().to_string(),
                "commit_intent": out.join("commit-intent.json").display().to_string(),
            }))
        }
        "proof" => {
            let b: Bundle = serde_json::from_slice(&std::fs::read(need(&m, "bundle")?)?)?;
            check_bundle_roots(&b)?;
            let p = chunk_proof(&b, &PathBuf::from(need(&m, "deltas")?), num(&m, "chunk")?)?;
            print(&p)
        }
        "check-bundle" => {
            let b: Bundle = serde_json::from_slice(&std::fs::read(need(&m, "bundle")?)?)?;
            check_bundle_roots(&b)?;
            print(&serde_json::json!({"ok": true, "record_digest": hex0x(&b.record_digest)}))
        }
        "roots" => {
            let b: Bundle = serde_json::from_slice(&std::fs::read(need(&m, "bundle")?)?)?;
            print(&recompute_roots(&b)?)
        }
        "intent" => {
            let b: Bundle = serde_json::from_slice(&std::fs::read(need(&m, "bundle")?)?)?;
            check_bundle_roots(&b)?;
            print(&commit_intent(&b))
        }
        other => bail!("unknown subcommand {other:?}"),
    }
}
