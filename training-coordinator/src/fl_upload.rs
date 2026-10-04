//! `PUT /v1/fl/delta/{sha256}?job=<id>`: where workers place federated LoRA
//! delta artifacts before submitting their signed result.
//!
//! Off unless the operator configures a delta directory
//! (`CITRATE_COORDINATOR_FL_DELTA_DIR`); a coordinator that does not run LoRA
//! rounds serves no upload route at all.
//!
//! An upload is accepted only when all of these hold, checked before a byte of
//! the body is stored:
//!
//! * the `x-citrate-signature` header recovers, over
//!   [`fl_delta_upload_digest`]`(job, sha256)`, to the worker currently holding an
//!   unexpired lease on `job`;
//! * `job` is a `lora_delta` job whose payload validates.
//!
//! The body is then streamed to a temporary file with a running SHA-256 and a
//! byte cap (the round's `max_values` bound, or the operator cap if lower). It is
//! kept, under its content address, only if it hashes to the path and its header
//! names this round, this worker and this round's start adapter. Anything else
//! is deleted. Re-uploading bytes already held is a no-op.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, Query, State as AxumState};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::put;
use axum::{Json, Router};
use citrate_training_worker::coordinator_protocol::{
    fl_delta_upload_digest, FL_DELTA_SIGNATURE_HEADER,
};
use citrate_training_worker::fl::delta::{Artifact, HEADER_LEN};
use citrate_training_worker::fl::round::LoraDeltaPayload;
use citrate_training_worker::wallet::Wallet;
use futures::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::api::{Coordinator, ErrorBody};
use crate::job::JobId;
use crate::state::JobStatus;

/// Default operator cap on one delta artifact (placeholder, pending owner
/// sign-off): 256 MiB.
pub const DEFAULT_MAX_DELTA_BYTES: u64 = 256 << 20;

/// Where accepted deltas live, and the operator's per-artifact cap.
#[derive(Debug, Clone)]
pub struct DeltaStore {
    dir: PathBuf,
    max_bytes: u64,
}

impl DeltaStore {
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, max_bytes })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// `<dir>/<sha256 hex>.fld`.
    pub fn path_of(&self, sha: &[u8; 32]) -> PathBuf {
        self.dir.join(format!("{}.fld", hex::encode(sha)))
    }

    /// A fresh temporary path for one upload of `sha`: unique within this process (a counter)
    /// and across processes sharing the directory (the process id).
    fn partial_path(&self, sha: &[u8; 32]) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        self.dir.join(format!(
            ".{}.{}.{n}.partial",
            hex::encode(sha),
            std::process::id()
        ))
    }
}

#[derive(Deserialize)]
pub struct UploadQuery {
    job: String,
}

type ApiError = (StatusCode, Json<ErrorBody>);

fn err(code: StatusCode, msg: impl Into<String>) -> ApiError {
    (code, Json(ErrorBody { error: msg.into() }))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The upload route, to be merged into the coordinator's router.
pub fn fl_router(c: Arc<Coordinator>, store: DeltaStore) -> Router {
    Router::new()
        .route("/v1/fl/delta/:sha", put(upload))
        // The body is streamed under our own cap; axum's 2 MB default would
        // refuse every real delta before the handler could size it properly.
        .layer(DefaultBodyLimit::disable())
        .with_state((c, Arc::new(store)))
}

fn parse_sha(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    hex::decode(s).ok()?.try_into().ok()
}

async fn upload(
    AxumState((c, store)): AxumState<(Arc<Coordinator>, Arc<DeltaStore>)>,
    UrlPath(sha_hex): UrlPath<String>,
    Query(q): Query<UploadQuery>,
    headers: HeaderMap,
    body: Body,
) -> Result<StatusCode, ApiError> {
    let sha = parse_sha(&sha_hex)
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "path must be a 64-hex sha256"))?;
    let sig_hex = headers
        .get(FL_DELTA_SIGNATURE_HEADER)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "missing upload signature"))?;
    let sig = hex::decode(sig_hex.trim_start_matches("0x"))
        .map_err(|_| err(StatusCode::UNAUTHORIZED, "signature is not hex"))?;
    let job = JobId(q.job);
    let who = Wallet::recover_address(&fl_delta_upload_digest(&job, &sha), &sig)
        .map_err(|_| err(StatusCode::UNAUTHORIZED, "signature does not recover"))?;

    // Authorisation: the recovered signer holds a live lease on a LoRA job.
    let snapshot = c.snapshot();
    let rec = snapshot
        .jobs
        .get(&job)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown job"))?;
    match rec.status {
        JobStatus::Leased {
            worker, expires_at, ..
        } if worker == who && expires_at > unix_now() => {}
        _ => {
            return Err(err(
                StatusCode::CONFLICT,
                "only the worker holding a live lease on this job may upload for it",
            ))
        }
    }
    let payload: LoraDeltaPayload =
        serde_json::from_value(rec.spec.payload.clone()).map_err(|_| {
            err(
                StatusCode::CONFLICT,
                "this job is not a federated LoRA round",
            )
        })?;
    payload
        .validate()
        .map_err(|e| err(StatusCode::CONFLICT, format!("round payload: {e}")))?;
    let cfg = &payload.config;
    let round_cap = cfg
        .max_values
        .checked_mul(8)
        .and_then(|b| b.checked_add(HEADER_LEN as u64))
        .unwrap_or(u64::MAX);
    let cap = round_cap.min(store.max_bytes);

    let dest = store.path_of(&sha);
    if dest.is_file() {
        return Ok(StatusCode::OK);
    }

    // Stream to a temp file beside the destination, hashing and counting. The name is unique
    // per request: two uploads of one address in flight at once must never share a file, or a
    // slow one could write its tail into the artifact the other already stored.
    let tmp = store.partial_path(&sha);
    let result = stream_to(&tmp, body, cap).await;
    let (got, len) = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    let reject = |code: StatusCode, msg: String| {
        let _ = std::fs::remove_file(&tmp);
        Err(err(code, msg))
    };
    if got != sha {
        return reject(
            StatusCode::UNPROCESSABLE_ENTITY,
            "the body does not hash to the path".into(),
        );
    }
    if len < HEADER_LEN as u64 {
        return reject(
            StatusCode::UNPROCESSABLE_ENTITY,
            "not a delta artifact".into(),
        );
    }
    // Header binding: read the artifact back (bounded by the cap above).
    let bytes = match tokio::fs::read(&tmp).await {
        Ok(b) => b,
        Err(e) => return reject(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let art = match Artifact::decode(&bytes, cfg.max_values) {
        Ok(a) => a,
        Err(e) => return reject(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()),
    };
    if art.round_id != payload.round_id
        || art.worker != who.to_fixed_bytes()
        || art.start_adapter_sha256 != cfg.start_adapter_sha256
        || art.chunk_dim != cfg.chunk_dim
        || art.value_scale_log2 != cfg.value_scale_log2
    {
        return reject(
            StatusCode::UNPROCESSABLE_ENTITY,
            "the artifact is not this worker's delta for this round".into(),
        );
    }
    if let Err(e) = tokio::fs::rename(&tmp, &dest).await {
        return reject(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    if let Ok(d) = tokio::fs::File::open(&store.dir).await {
        let _ = d.sync_all().await;
    }
    tracing::info!(job = %job, worker = ?who, bytes = len, sha = %hex::encode(sha), "fl delta stored");
    Ok(StatusCode::CREATED)
}

async fn stream_to(tmp: &Path, body: Body, cap: u64) -> Result<([u8; 32], u64), ApiError> {
    let mut f = tokio::fs::File::create(tmp)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut h = Sha256::new();
    let mut len = 0u64;
    let mut stream = body.into_data_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
        len = len.saturating_add(chunk.len() as u64);
        if len > cap {
            return Err(err(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("delta exceeds {cap} bytes"),
            ));
        }
        h.update(&chunk);
        f.write_all(&chunk)
            .await
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    f.sync_all()
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    Ok((out, len))
}
