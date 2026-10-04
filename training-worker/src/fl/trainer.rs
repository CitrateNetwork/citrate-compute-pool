//! The LoRA trainer seam.
//!
//! Training a LoRA on a GGUF base model is not something this crate does
//! itself: the device runs whatever trainer its operator installed (a PEFT
//! script on a CUDA box, an MLX script on Apple silicon) as an external program.
//! The worker's job is to hand it verified inputs and to verify what comes back.
//!
//! The program is configured by the operator (`CITRATE_LORA_TRAINER`), never by
//! the job: a coordinator cannot make a member's machine run a program of its
//! choosing. Inputs are passed as environment variables, not interpolated into
//! a command line, so no path or value can be read as a flag or a shell word:
//!
//! | Variable | Meaning |
//! |---|---|
//! | `CITRATE_LORA_BASE_MODEL` | base model GGUF (SHA-256 verified) |
//! | `CITRATE_LORA_START_ADAPTER` | adapter to start from (SHA-256 verified) |
//! | `CITRATE_LORA_DATASET` | verified trajectory export, JSONL |
//! | `CITRATE_LORA_OUT` | where the trained adapter must be written |
//! | `CITRATE_LORA_ROUND_ID` | the round, `0x` hex (a seed source) |
//!
//! The program must write a GGUF LoRA adapter with the same tensors as the start
//! adapter to `CITRATE_LORA_OUT` and exit 0 within the timeout.

use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct TrainRequest {
    pub base_model: PathBuf,
    pub start_adapter: PathBuf,
    pub dataset: PathBuf,
    pub out: PathBuf,
    pub round_id_hex: String,
}

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("could not start the trainer {program}: {source}")]
    Spawn {
        program: String,
        source: std::io::Error,
    },
    #[error("the trainer ran past its {0:?} timeout and was stopped")]
    Timeout(Duration),
    #[error("the trainer exited with {status}; stderr tail: {stderr}")]
    Failed { status: String, stderr: String },
    #[error("the trainer exited 0 but wrote no adapter at {0}")]
    NoOutput(PathBuf),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Something that turns a start adapter and a dataset into a trained adapter.
#[async_trait::async_trait]
pub trait LoraTrainer: Send + Sync {
    /// Provenance string recorded in the signed result.
    fn id(&self) -> String;
    async fn train(&self, req: &TrainRequest) -> Result<(), TrainError>;
}

/// Runs the operator's trainer program.
#[derive(Debug, Clone)]
pub struct CommandTrainer {
    program: PathBuf,
    timeout: Duration,
    /// SHA-256 of the program file at configuration time, for provenance.
    program_sha256: String,
}

/// Default ceiling on one training run (placeholder, pending owner sign-off).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(6 * 3600);
const STDERR_TAIL: usize = 2048;

impl CommandTrainer {
    pub fn new(program: impl Into<PathBuf>, timeout: Duration) -> std::io::Result<Self> {
        let program = program.into();
        let meta = std::fs::metadata(&program)?;
        if !meta.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is not a file", program.display()),
            ));
        }
        let program_sha256 = hex::encode(super::sha256_file(&program)?);
        Ok(Self {
            program,
            timeout,
            program_sha256,
        })
    }

    /// From `CITRATE_LORA_TRAINER` (and `CITRATE_LORA_TRAINER_TIMEOUT_SECS`).
    /// `Ok(None)` when no trainer is configured: the device then declines
    /// federated LoRA work with that reason.
    pub fn from_env() -> std::io::Result<Option<Self>> {
        let Ok(p) = std::env::var("CITRATE_LORA_TRAINER") else {
            return Ok(None);
        };
        if p.trim().is_empty() {
            return Ok(None);
        }
        let timeout = match std::env::var("CITRATE_LORA_TRAINER_TIMEOUT_SECS") {
            Ok(s) => Duration::from_secs(s.trim().parse().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("CITRATE_LORA_TRAINER_TIMEOUT_SECS={s:?}: {e}"),
                )
            })?),
            Err(_) => DEFAULT_TIMEOUT,
        };
        Self::new(PathBuf::from(p.trim()), timeout).map(Some)
    }

    pub fn program(&self) -> &Path {
        &self.program
    }
}

#[async_trait::async_trait]
impl LoraTrainer for CommandTrainer {
    fn id(&self) -> String {
        let name = self
            .program
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{name}@sha256:{}", self.program_sha256)
    }

    async fn train(&self, req: &TrainRequest) -> Result<(), TrainError> {
        if req.out.exists() {
            std::fs::remove_file(&req.out)?;
        }
        let mut cmd = tokio::process::Command::new(&self.program);
        cmd.env("CITRATE_LORA_BASE_MODEL", &req.base_model)
            .env("CITRATE_LORA_START_ADAPTER", &req.start_adapter)
            .env("CITRATE_LORA_DATASET", &req.dataset)
            .env("CITRATE_LORA_OUT", &req.out)
            .env("CITRATE_LORA_ROUND_ID", &req.round_id_hex)
            // The trainer never sees the worker's key material.
            .env_remove("CITRATE_TRAINING_PRIVATE_KEY_HEX")
            .env_remove("CITRATE_TRAINING_KEYSTORE_PASSPHRASE")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        let child = cmd.spawn().map_err(|source| TrainError::Spawn {
            program: self.program.display().to_string(),
            source,
        })?;
        let out = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(r) => r?,
            Err(_) => return Err(TrainError::Timeout(self.timeout)),
        };
        if !out.status.success() {
            let s = String::from_utf8_lossy(&out.stderr);
            let tail: String = s
                .chars()
                .rev()
                .take(STDERR_TAIL)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(TrainError::Failed {
                status: out.status.to_string(),
                stderr: tail,
            });
        }
        if !req.out.is_file() {
            return Err(TrainError::NoOutput(req.out.clone()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(dir).expect("dir");
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).expect("write");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
        p
    }

    fn req(dir: &Path) -> TrainRequest {
        TrainRequest {
            base_model: dir.join("base.gguf"),
            start_adapter: dir.join("start.gguf"),
            dataset: dir.join("data.jsonl"),
            out: dir.join("out.gguf"),
            round_id_hex: "0xabc".into(),
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("fl-trainer-{tag}-{}", std::process::id()))
    }

    #[tokio::test]
    async fn inputs_arrive_as_environment_and_the_output_is_checked() {
        let dir = tmp("env");
        let p = script(
            &dir,
            "t.sh",
            "printf '%s|%s|%s|%s' \"$CITRATE_LORA_BASE_MODEL\" \"$CITRATE_LORA_START_ADAPTER\" \
             \"$CITRATE_LORA_DATASET\" \"$CITRATE_LORA_ROUND_ID\" > \"$CITRATE_LORA_OUT\"",
        );
        let t = CommandTrainer::new(&p, Duration::from_secs(30)).expect("trainer");
        let r = req(&dir);
        t.train(&r).await.expect("train");
        let got = std::fs::read_to_string(&r.out).expect("out");
        assert_eq!(
            got,
            format!(
                "{}|{}|{}|0xabc",
                r.base_model.display(),
                r.start_adapter.display(),
                r.dataset.display()
            )
        );
        assert!(t.id().starts_with("t.sh@sha256:"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_failing_trainer_reports_its_stderr() {
        let dir = tmp("fail");
        let p = script(&dir, "t.sh", "echo 'out of memory' >&2; exit 3");
        let t = CommandTrainer::new(&p, Duration::from_secs(30)).expect("trainer");
        match t.train(&req(&dir)).await {
            Err(TrainError::Failed { stderr, .. }) => assert!(stderr.contains("out of memory")),
            other => panic!("expected Failed, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn exiting_zero_without_an_adapter_is_a_failure() {
        let dir = tmp("noout");
        let p = script(&dir, "t.sh", "exit 0");
        let t = CommandTrainer::new(&p, Duration::from_secs(30)).expect("trainer");
        assert!(matches!(
            t.train(&req(&dir)).await,
            Err(TrainError::NoOutput(_))
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_stale_output_from_an_earlier_run_does_not_count() {
        let dir = tmp("stale");
        let p = script(&dir, "t.sh", "exit 0");
        let t = CommandTrainer::new(&p, Duration::from_secs(30)).expect("trainer");
        let r = req(&dir);
        std::fs::write(&r.out, b"old").expect("stale");
        assert!(matches!(t.train(&r).await, Err(TrainError::NoOutput(_))));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_hung_trainer_is_stopped_at_the_timeout() {
        let dir = tmp("hang");
        let p = script(&dir, "t.sh", "sleep 30");
        let t = CommandTrainer::new(&p, Duration::from_millis(300)).expect("trainer");
        assert!(matches!(
            t.train(&req(&dir)).await,
            Err(TrainError::Timeout(_))
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_program_is_a_configuration_error() {
        assert!(CommandTrainer::new("/nonexistent/trainer", DEFAULT_TIMEOUT).is_err());
    }
}
