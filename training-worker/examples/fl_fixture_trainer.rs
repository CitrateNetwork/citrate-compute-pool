//! `fl_fixture_trainer`: NOT A TRAINER. A stand-in for the operator's LoRA trainer
//! program, used only by the local devnet round end-to-end
//! (citrate-chain `scripts/fl/devnet-round-e2e.sh`) to exercise the round
//! protocol on a machine that has no LoRA training stack.
//!
//! It follows the trainer contract in `fl::trainer` exactly (inputs from the
//! `CITRATE_LORA_*` environment, a GGUF LoRA adapter with the start adapter's
//! tensors written to `CITRATE_LORA_OUT`), but instead of learning anything it
//! moves every `lora_b` value by a deterministic amount derived from the
//! dataset's SHA-256 and the round id. Different datasets therefore produce
//! different deltas (including opposite signs), which is what the aggregation,
//! commitment, replay and challenge paths need to be exercised. No learning
//! happens, and the end-to-end receipt says so.
//!
//! Real rounds use a real trainer (for example a PEFT or MLX script on the
//! device) configured by the operator through `CITRATE_LORA_TRAINER`.

use std::path::PathBuf;

use citrate_training_worker::fl::gguf::{encode_f32, Adapter, OutTensor};
use citrate_training_worker::fl::sha256;

fn env(k: &str) -> Result<String, String> {
    std::env::var(k).map_err(|_| format!("{k} is not set"))
}

fn run() -> Result<(), String> {
    let start = PathBuf::from(env("CITRATE_LORA_START_ADAPTER")?);
    let dataset = std::fs::read(env("CITRATE_LORA_DATASET")?).map_err(|e| e.to_string())?;
    let round = env("CITRATE_LORA_ROUND_ID")?;
    let out = PathBuf::from(env("CITRATE_LORA_OUT")?);
    let a = Adapter::load(&start).map_err(|e| e.to_string())?;
    let seed = sha256(&[sha256(&dataset).as_slice(), round.as_bytes()].concat());
    let mut values = a.values.clone();
    for (ti, (t, v)) in a.header.tensors.iter().zip(values.iter_mut()).enumerate() {
        if !t.name.ends_with(".lora_b") {
            continue;
        }
        for (i, x) in v.iter_mut().enumerate() {
            // A small, sign-varying step per value: in (-0.004, 0.004).
            let b = seed[(ti + i) % 32];
            let step = (f32::from(b) - 127.5) / 127.5 * 0.004;
            *x += step;
        }
    }
    let tensors: Vec<OutTensor<'_>> = a
        .header
        .tensors
        .iter()
        .zip(values.iter())
        .map(|(t, v)| OutTensor {
            name: &t.name,
            dims: &t.dims,
            values: v,
        })
        .collect();
    let bytes = encode_f32(&a.header.kv, &tensors).map_err(|e| e.to_string())?;
    std::fs::write(&out, bytes).map_err(|e| e.to_string())?;
    eprintln!("fl_fixture_trainer: wrote a fixture adapter (no training ran)");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("fl_fixture_trainer: {e}");
        std::process::exit(1);
    }
}
