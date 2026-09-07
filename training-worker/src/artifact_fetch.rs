//! Getting artifacts onto a member's machine, from a mirror nobody has to trust.
//!
//! ## The mirror is untrusted, and that is the whole design
//!
//! Every artifact this fetches is content-addressed, and every one is verified
//! against a hash that came from somewhere else:
//!
//! | artifact | verified against | committed by |
//! |---|---|---|
//! | checkpoint | `model_start_hash` | the job |
//! | manifest | `dataset_hash` | the job |
//! | shard | `provenance_root` | the **verified** manifest |
//!
//! So a mirror can be a CDN, a droplet, a peer, or a stranger's S3 bucket. It
//! cannot substitute a checkpoint, poison a corpus, or slip in a shard the
//! manifest does not commit to — the worst it can do is fail to serve, and a
//! worker that cannot get artifacts declines the job, which is already a
//! first-class outcome.
//!
//! That is why this is plain HTTP with no signatures, no keys and no TLS pinning:
//! adding them would protect a property that content addressing already
//! guarantees, while adding a key-distribution problem that content addressing
//! does not have.
//!
//! ## Nothing unverified is ever placed in the store
//!
//! Every fetch downloads to a temp file, hashes it, compares, and only then
//! renames into place. A hash mismatch leaves the store exactly as it was. This
//! matters more than it sounds: `ArtifactStore::resolve` trusts what is on disk,
//! so a half-written or wrong-hash file left behind would be verified once,
//! rejected, and then re-fetched forever — or worse, a truncated file that
//! happens to parse.
//!
//! ## Sizes, measured
//!
//! corpus-v6 is 2.4 GB in total, and fetching all of it would be the obvious
//! design and the wrong one. A worker reads only the shards
//! [`crate::nat_backend::shard_slice_for`] selects — a few per step, ~7.4 KB
//! each. What a job actually needs is the 80 MB manifest, the 123 MB checkpoint,
//! and single-digit MB of shards: roughly 200 MB rather than 2.4 GB.

use std::path::{Path, PathBuf};

use ethereum_types::H256;
use sha3::{Digest, Keccak256};

/// Cap on any single artifact, so a hostile or broken mirror cannot stream a
/// volunteer's disk full. The largest real artifact is the 123 MB checkpoint;
/// 1 GiB leaves generous headroom for a bigger model without leaving the door
/// open.
pub const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("mirror is unreachable: {0}")]
    Transport(String),
    #[error("mirror returned {status} for {path}")]
    NotServed { status: u16, path: String },
    #[error(
        "content hash mismatch for {path}: wanted {wanted:?}, got {got:?}. \
         The mirror served something other than what the job named; nothing was written."
    )]
    HashMismatch {
        path: String,
        wanted: H256,
        got: H256,
    },
    #[error("artifact exceeds the {MAX_ARTIFACT_BYTES}-byte cap")]
    TooLarge,
    #[error("io: {0}")]
    Io(String),
}

/// A source of content-addressed artifacts.
pub trait ArtifactSource {
    /// Fetch `path` and return its bytes. Implementations MUST NOT verify —
    /// verification is [`fetch_verified`]'s job, so there is exactly one place it
    /// can be forgotten.
    fn get(
        &self,
        path: &str,
    ) -> impl std::future::Future<Output = Result<Vec<u8>, FetchError>> + Send;
}

/// An HTTP mirror. Paths mirror the [`crate::job_artifacts::ArtifactStore`]
/// layout, so a mirror can literally be that directory served statically —
/// `python3 -m http.server` over an artifact store is a valid mirror, which makes
/// a member able to seed for a peer with no software at all.
pub struct HttpMirror {
    base: String,
    http: reqwest::Client,
}

impl HttpMirror {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            base: base.into().trim_end_matches('/').to_string(),
            // CP-B-006: redirect-safe + timeout-bounded (a bare
            // `Client::new()` follows redirects and never times out).
            http: crate::outbound::redirect_safe_client(std::time::Duration::from_secs(120)),
        }
    }
}

impl HttpMirror {
    /// Fetch `path`, refusing anything larger than `max_bytes`.
    ///
    /// CP-B-011: the cap is enforced by STREAMING the body with a running
    /// byte counter and aborting the instant it is exceeded — not by
    /// buffering the whole response and checking its length afterwards. A
    /// chunked response with no `Content-Length` (which the untrusted
    /// mirror fully controls) could otherwise stream unboundedly into RAM
    /// and OOM the volunteer's process before a post-`bytes()` check could
    /// ever fire. `Response::chunk()` is available without reqwest's
    /// `stream` feature.
    pub(crate) async fn fetch_capped(
        &self,
        path: &str,
        max_bytes: u64,
    ) -> Result<Vec<u8>, FetchError> {
        let res = self
            .http
            .get(format!("{}/{path}", self.base))
            .send()
            .await
            .map_err(|e| FetchError::Transport(e.to_string()))?;
        let status = res.status();
        if !status.is_success() {
            return Err(FetchError::NotServed {
                status: status.as_u16(),
                path: path.to_string(),
            });
        }
        // Refuse on the advertised length before reading a byte where possible;
        // the streaming check below is what actually enforces it, since
        // Content-Length is a claim and not a promise.
        if let Some(len) = res.content_length() {
            if len > max_bytes {
                return Err(FetchError::TooLarge);
            }
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut res = res;
        while let Some(chunk) = res
            .chunk()
            .await
            .map_err(|e| FetchError::Transport(e.to_string()))?
        {
            if buf.len() as u64 + chunk.len() as u64 > max_bytes {
                return Err(FetchError::TooLarge);
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf)
    }
}

impl ArtifactSource for HttpMirror {
    async fn get(&self, path: &str) -> Result<Vec<u8>, FetchError> {
        self.fetch_capped(path, MAX_ARTIFACT_BYTES).await
    }
}

pub fn keccak256(bytes: &[u8]) -> H256 {
    let mut h = Keccak256::new();
    h.update(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    H256(out)
}

/// Write `bytes` to `dest` only if they hash to `wanted`.
///
/// Verify-then-place, via a temp file in the same directory and an atomic rename.
/// On mismatch the destination is untouched and the temp file is removed, so a
/// failed fetch can never leave the store in a state `resolve` would trust.
pub fn place_verified(
    dest: &Path,
    bytes: &[u8],
    wanted: H256,
    label: &str,
) -> Result<(), FetchError> {
    let got = keccak256(bytes);
    if got != wanted {
        return Err(FetchError::HashMismatch {
            path: label.to_string(),
            wanted,
            got,
        });
    }
    let dir = dest
        .parent()
        .ok_or_else(|| FetchError::Io("destination has no parent".into()))?;
    std::fs::create_dir_all(dir).map_err(|e| FetchError::Io(e.to_string()))?;

    // Same directory, so the rename stays within one filesystem and is atomic.
    let tmp = dest.with_extension("partial");
    std::fs::write(&tmp, bytes).map_err(|e| FetchError::Io(e.to_string()))?;
    std::fs::rename(&tmp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        FetchError::Io(e.to_string())
    })?;
    Ok(())
}

/// Fetch and place in one step, skipping the transfer entirely if the file is
/// already present AND already hashes correctly.
///
/// Re-hashing a present file rather than trusting its existence is deliberate: a
/// store can be edited, a disk can rot, and a previous run can have been killed
/// mid-write by something other than this code.
pub async fn fetch_verified<S: ArtifactSource>(
    source: &S,
    remote_path: &str,
    dest: &Path,
    wanted: H256,
) -> Result<bool, FetchError> {
    if let Ok(existing) = std::fs::read(dest) {
        if keccak256(&existing) == wanted {
            return Ok(false); // already have it
        }
    }
    let bytes = source.get(remote_path).await?;
    place_verified(dest, &bytes, wanted, remote_path)?;
    Ok(true)
}

/// The store-relative path of a model file, matching `ArtifactStore::model_dir`.
pub fn model_path(hash: &H256, file: &str) -> String {
    format!("models/0x{}/{file}", hex::encode(hash.as_bytes()))
}

/// The store-relative path of a dataset file, matching `ArtifactStore::dataset_dir`.
pub fn dataset_path(hash: &H256, file: &str) -> String {
    format!("datasets/0x{}/{file}", hex::encode(hash.as_bytes()))
}

/// Local destination for a store-relative path.
///
/// Rejects any traversal component. The path components here are derived from
/// hex hashes and a fixed file name so traversal should be impossible — but this
/// turns "should be impossible" into "is checked", and it is the difference
/// between a mirror serving a bad file and a mirror writing outside the store.
pub fn local_dest(store_root: &Path, rel: &str) -> Result<PathBuf, FetchError> {
    for part in rel.split('/') {
        if part.is_empty() || part == "." || part == ".." || part.contains('\\') {
            return Err(FetchError::Io(format!(
                "refusing unsafe artifact path {rel:?}"
            )));
        }
    }
    Ok(store_root.join(rel))
}

#[cfg(test)]
mod tests {
    include!("artifact_fetch_tests.rs");
}
