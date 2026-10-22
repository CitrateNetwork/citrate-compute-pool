//! Runs `scripts/ci/check_wallet_zeroize.sh` from the test suite, so the check
//! that both wallet copies keep their decrypted key material in `Zeroizing`
//! fails `cargo test`, not only a script nobody runs.
//!
//! The negative control copies the two wallet files into a temp tree, unwraps
//! one intermediate, and requires the script to fail on it. The script only
//! reads files; the temp tree is the only thing written.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn run_check(tree: &Path) -> (bool, String) {
    let script = repo_root().join("scripts/ci/check_wallet_zeroize.sh");
    let out = Command::new("bash")
        .arg(&script)
        .arg(tree)
        .output()
        .expect("run bash");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

fn copy_wallets_to(dir: &Path) {
    for rel in [
        "training-worker/src/wallet.rs",
        "pool-coordinator/src/wallet.rs",
    ] {
        let dst = dir.join(rel);
        std::fs::create_dir_all(dst.parent().expect("parent")).expect("mkdir");
        std::fs::copy(repo_root().join(rel), &dst).expect("copy wallet");
    }
}

#[test]
fn both_wallets_keep_key_material_zeroized() {
    let (ok, text) = run_check(&repo_root());
    assert!(ok, "wallet zeroize tripwire failed on this tree:\n{text}");
}

#[test]
fn the_tripwire_catches_each_unwrapped_intermediate() {
    let cases = [
        (
            "training-worker/src/wallet.rs",
            "let mut derived = Zeroizing::new([0u8; 32]);",
            "let mut derived = [0u8; 32];",
        ),
        (
            "pool-coordinator/src/wallet.rs",
            "let mut sk_bytes = Zeroizing::new([0u8; 32]);",
            "let mut sk_bytes = [0u8; 32];",
        ),
        (
            "pool-coordinator/src/wallet.rs",
            "let hex_key = Zeroizing::new(hex::encode(&plaintext[..]));",
            "let hex_key = hex::encode(&plaintext[..]);",
        ),
    ];
    for (i, (rel, wrapped, bare)) in cases.iter().enumerate() {
        let dir = std::env::temp_dir().join(format!(
            "citrate-zeroize-tripwire-{}-{i}",
            std::process::id()
        ));
        if dir.starts_with(std::env::temp_dir()) && dir.exists() {
            std::fs::remove_dir_all(&dir).expect("clear temp tree");
        }
        copy_wallets_to(&dir);
        let f = dir.join(rel);
        let src = std::fs::read_to_string(&f).expect("read copy");
        assert_eq!(src.matches(wrapped).count(), 1, "{rel}: fixture drifted");
        std::fs::write(&f, src.replace(wrapped, bare)).expect("write copy");
        let (ok, text) = run_check(&dir);
        if dir.starts_with(std::env::temp_dir()) {
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert!(
            !ok,
            "unwrapping `{wrapped}` in {rel} must fail the tripwire:\n{text}"
        );
    }
}
