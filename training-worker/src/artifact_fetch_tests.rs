// Fetching from a mirror nobody trusts.
//
// The property under test throughout: after ANY failure, the store must be in a
// state `ArtifactStore::resolve` would either accept as correct or treat as
// absent — never a third state where a wrong or partial file sits where a right
// one belongs. `resolve` trusts the disk, so the disk is what has to be right.

use super::*;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("citrate-fetch-test-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A mirror that serves exactly what it is told to, including lies.
struct StubMirror {
    body: Vec<u8>,
    fail: Option<FetchError>,
    hits: std::sync::atomic::AtomicUsize,
}

impl StubMirror {
    fn serving(body: &[u8]) -> Self {
        Self {
            body: body.to_vec(),
            fail: None,
            hits: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    fn failing(e: FetchError) -> Self {
        Self {
            body: vec![],
            fail: Some(e),
            hits: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    fn hits(&self) -> usize {
        self.hits.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl ArtifactSource for StubMirror {
    async fn get(&self, _path: &str) -> Result<Vec<u8>, FetchError> {
        self.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match &self.fail {
            Some(FetchError::TooLarge) => Err(FetchError::TooLarge),
            Some(FetchError::Transport(m)) => Err(FetchError::Transport(m.clone())),
            Some(FetchError::NotServed { status, path }) => Err(FetchError::NotServed {
                status: *status,
                path: path.clone(),
            }),
            Some(_) | None if self.fail.is_some() => Err(FetchError::Transport("stub".into())),
            _ => Ok(self.body.clone()),
        }
    }
}

// ── The happy path ─────────────────────────────────────────────────────

#[tokio::test]
async fn a_correct_artifact_is_placed() {
    let d = tmpdir("happy");
    let body = b"the real checkpoint bytes";
    let m = StubMirror::serving(body);
    let dest = d.join("models/0xabc/model.safetensors");

    let fetched = fetch_verified(&m, "models/0xabc/model.safetensors", &dest, keccak256(body))
        .await
        .unwrap();
    assert!(fetched);
    assert_eq!(std::fs::read(&dest).unwrap(), body);
}

/// Re-hashing a present file rather than trusting its existence: a store can be
/// edited, a disk can rot, a previous run can have died mid-write.
#[tokio::test]
async fn an_artifact_already_present_and_correct_is_not_refetched() {
    let d = tmpdir("cached");
    let body = b"already here";
    let dest = d.join("models/0xabc/model.safetensors");
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(&dest, body).unwrap();

    let m = StubMirror::serving(body);
    let fetched = fetch_verified(&m, "p", &dest, keccak256(body)).await.unwrap();
    assert!(!fetched, "should not have transferred");
    assert_eq!(m.hits(), 0, "the mirror should not have been contacted");
}

/// A file that is present but WRONG must be re-fetched, not trusted for existing.
#[tokio::test]
async fn a_present_but_corrupt_artifact_is_replaced() {
    let d = tmpdir("corrupt");
    let good = b"the right bytes";
    let dest = d.join("models/0xabc/model.safetensors");
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(&dest, b"rotted").unwrap();

    let m = StubMirror::serving(good);
    assert!(fetch_verified(&m, "p", &dest, keccak256(good)).await.unwrap());
    assert_eq!(std::fs::read(&dest).unwrap(), good);
}

// ── What an untrusted mirror cannot do ─────────────────────────────────

/// The property the whole design rests on. A mirror serving something other than
/// what the job named must not get it into the store.
#[tokio::test]
async fn a_mirror_serving_the_wrong_bytes_is_rejected_and_writes_nothing() {
    let d = tmpdir("wrongbytes");
    let dest = d.join("models/0xabc/model.safetensors");
    let m = StubMirror::serving(b"a poisoned checkpoint");

    let e = fetch_verified(&m, "p", &dest, keccak256(b"what the job asked for"))
        .await
        .unwrap_err();
    assert!(matches!(e, FetchError::HashMismatch { .. }), "got {e:?}");
    assert!(!dest.exists(), "nothing may be left where a good file belongs");
}

/// And it must not leave a `.partial` behind either — a stray temp file is
/// litter at best, and at worst something a later change reads.
#[tokio::test]
async fn a_rejected_fetch_leaves_no_temp_file() {
    let d = tmpdir("notmp");
    let dest = d.join("models/0xabc/model.safetensors");
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    let m = StubMirror::serving(b"wrong");
    let _ = fetch_verified(&m, "p", &dest, keccak256(b"right")).await;

    let left: Vec<_> = std::fs::read_dir(dest.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(left.is_empty(), "left behind: {left:?}");
}

/// An existing GOOD file must survive a later bad fetch. Otherwise a hostile
/// mirror could destroy a member's staged corpus just by serving garbage.
#[tokio::test]
async fn a_bad_fetch_does_not_destroy_a_good_file_already_in_place() {
    let d = tmpdir("preserve");
    let good = b"the good artifact";
    let dest = d.join("models/0xabc/model.safetensors");
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(&dest, good).unwrap();

    // Ask for a DIFFERENT hash, which the mirror answers with junk.
    let m = StubMirror::serving(b"junk");
    let _ = fetch_verified(&m, "p", &dest, keccak256(b"some other artifact")).await;
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        good,
        "the previously-good file must be intact"
    );
}

#[tokio::test]
async fn an_unreachable_mirror_is_an_error_not_a_panic() {
    let d = tmpdir("down");
    let m = StubMirror::failing(FetchError::Transport("connection refused".into()));
    let e = fetch_verified(&m, "p", &d.join("x"), H256::zero()).await.unwrap_err();
    assert!(matches!(e, FetchError::Transport(_)));
}

#[tokio::test]
async fn an_oversized_artifact_is_refused() {
    let d = tmpdir("toobig");
    let m = StubMirror::failing(FetchError::TooLarge);
    let e = fetch_verified(&m, "p", &d.join("x"), H256::zero()).await.unwrap_err();
    assert!(matches!(e, FetchError::TooLarge));
}

// ── Path safety ────────────────────────────────────────────────────────

/// Paths are built from hex hashes so traversal should be impossible. This turns
/// "should be" into "is", and it is the difference between a mirror serving a bad
/// file and a mirror writing outside the store.
#[test]
fn traversal_components_are_refused() {
    let root = Path::new("/store");
    for bad in [
        "../etc/passwd",
        "models/../../etc/passwd",
        "models//model.safetensors",
        "models/./x",
        "models/0xabc\\..\\x",
    ] {
        assert!(local_dest(root, bad).is_err(), "{bad:?} must be refused");
    }
}

#[test]
fn ordinary_store_paths_resolve_under_the_root() {
    let root = Path::new("/store");
    let p = local_dest(root, "models/0xabc/model.safetensors").unwrap();
    assert!(p.starts_with(root));
    assert!(p.ends_with("models/0xabc/model.safetensors"));
}

/// The remote layout must match `ArtifactStore`'s local layout exactly, so a
/// member can serve their own store directory statically and be a valid mirror
/// with no software at all.
#[test]
fn remote_paths_mirror_the_local_store_layout() {
    let h = H256::repeat_byte(0xAB);
    let hex = format!("0x{}", hex::encode(h.as_bytes()));
    assert_eq!(model_path(&h, "model.safetensors"), format!("models/{hex}/model.safetensors"));
    assert_eq!(model_path(&h, "sidecar.nat.json"), format!("models/{hex}/sidecar.nat.json"));
    assert_eq!(dataset_path(&h, "manifest.json"), format!("datasets/{hex}/manifest.json"));
    assert_eq!(dataset_path(&h, "shard_0042.json"), format!("datasets/{hex}/shard_0042.json"));
}

#[test]
fn a_hash_mismatch_error_names_both_hashes_so_it_can_be_diagnosed() {
    let e = FetchError::HashMismatch {
        path: "models/0xabc/model.safetensors".into(),
        wanted: H256::repeat_byte(1),
        got: H256::repeat_byte(2),
    };
    let s = e.to_string();
    assert!(s.contains("nothing was written"));
    assert!(s.contains("model.safetensors"));
}

// ── CP-B-011: streaming byte cap on the untrusted mirror ────────────────

/// Serve a `Transfer-Encoding: chunked` HTTP/1.1 response with NO
/// `Content-Length`, streaming `total` bytes in small chunks. This is
/// the shape an untrusted mirror controls to defeat a pre-read length
/// check — the exact case CP-B-011 is about. Returns the bound address.
async fn spawn_chunked_oversize_server(total: usize) -> std::net::SocketAddr {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        if let Ok((mut sock, _)) = listener.accept().await {
            // Drain the request headers (best-effort).
            let mut scratch = [0u8; 1024];
            let _ = sock.read(&mut scratch).await;
            let _ = sock
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
                )
                .await;
            let chunk = [b'x'; 64];
            let mut sent = 0usize;
            while sent < total {
                let n = chunk.len().min(total - sent);
                let _ = sock
                    .write_all(format!("{:x}\r\n", n).as_bytes())
                    .await;
                let _ = sock.write_all(&chunk[..n]).await;
                let _ = sock.write_all(b"\r\n").await;
                sent += n;
            }
            let _ = sock.write_all(b"0\r\n\r\n").await;
            let _ = sock.flush().await;
        }
    });
    addr
}

// CP-B-011 RED→GREEN: a chunked mirror response with no Content-Length
// that exceeds the cap must be refused as TooLarge from the STREAMING
// counter, never buffered whole. Pre-fix `get` did `res.bytes().await`
// (buffering the entire body) before the size check, so an unbounded
// chunked body OOM'd the process. Post-fix `fetch_capped` aborts mid-
// stream the instant the running total crosses the cap.
#[tokio::test]
async fn a_chunked_oversize_mirror_response_is_refused_by_the_streaming_cap() {
    // Cap of 256 bytes; server streams 4 KiB with no Content-Length.
    let addr = spawn_chunked_oversize_server(4096).await;
    let mirror = HttpMirror::new(format!("http://{addr}"));
    let out = mirror.fetch_capped("anything", 256).await;
    assert!(
        matches!(out, Err(FetchError::TooLarge)),
        "an over-cap chunked body must be refused as TooLarge, got {out:?}"
    );
}

// The same path must still SERVE a body that fits under the cap.
#[tokio::test]
async fn a_chunked_response_under_the_cap_is_served() {
    let addr = spawn_chunked_oversize_server(100).await;
    let mirror = HttpMirror::new(format!("http://{addr}"));
    let out = mirror.fetch_capped("anything", 256).await.expect("under cap serves");
    assert_eq!(out.len(), 100);
}
