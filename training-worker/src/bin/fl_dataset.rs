//! `citrate-fl-dataset`: turn a member's verified trajectory export into trainer input rows
//! (HUP-S9.3, RA-15). See `citrate_training_worker::fl::dataset` for what is kept and dropped.
//!
//! ```text
//! citrate-fl-dataset --export <export.jsonl> --out <rows.jsonl> [--report <report.json>]
//!                    [--heldout <manifest.json>]
//! ```
//!
//! `--heldout` defaults to the embedded toolcall-v2 manifest. The output is created new (an
//! existing file is never overwritten), readable only by its owner on Unix, and is local: nothing
//! here uploads anything. Exit status 2 when no row survived, so a round never starts on an
//! empty training set.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use citrate_training_worker::fl::dataset::{convert_export, HeldOut};
use citrate_training_worker::fl::runner::MAX_DATASET_BYTES;

fn usage() -> String {
    "usage: citrate-fl-dataset --export <export.jsonl> --out <rows.jsonl> \
     [--report <report.json>] [--heldout <manifest.json>]"
        .into()
}

fn args() -> Result<BTreeMap<String, String>, String> {
    let mut m = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let key = k
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {k:?}\n{}", usage()))?;
        if !matches!(key, "export" | "out" | "report" | "heldout") {
            return Err(format!("unknown option --{key}\n{}", usage()));
        }
        let v = it
            .next()
            .ok_or_else(|| format!("--{key} needs a value\n{}", usage()))?;
        m.insert(key.to_string(), v);
    }
    Ok(m)
}

fn create_new(path: &PathBuf, body: &[u8]) -> Result<(), String> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(body)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<u8, String> {
    let m = args()?;
    let export = PathBuf::from(m.get("export").ok_or_else(usage)?);
    let out = PathBuf::from(m.get("out").ok_or_else(usage)?);
    let len = std::fs::metadata(&export)
        .map_err(|e| format!("{}: {e}", export.display()))?
        .len();
    if len > MAX_DATASET_BYTES {
        return Err(format!(
            "{} is {len} bytes; the device limit is {MAX_DATASET_BYTES}",
            export.display()
        ));
    }
    let bytes = std::fs::read(&export).map_err(|e| format!("{}: {e}", export.display()))?;
    let held = match m.get("heldout") {
        Some(p) => HeldOut::parse(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?),
        None => HeldOut::toolcall_v2(),
    }
    .map_err(|e| e.to_string())?;
    let converted = convert_export(&bytes, &held).map_err(|e| e.to_string())?;
    let report = serde_json::to_string_pretty(&converted.report).map_err(|e| e.to_string())?;
    if let Some(p) = m.get("report") {
        create_new(&PathBuf::from(p), format!("{report}\n").as_bytes())?;
    }
    println!("{report}");
    if converted.rows.is_empty() {
        eprintln!("citrate-fl-dataset: no example survived; nothing written");
        return Ok(2);
    }
    create_new(
        &out,
        converted.to_jsonl().map_err(|e| e.to_string())?.as_bytes(),
    )?;
    Ok(0)
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(e) => {
            eprintln!("citrate-fl-dataset: {e}");
            std::process::ExitCode::from(1)
        }
    }
}
