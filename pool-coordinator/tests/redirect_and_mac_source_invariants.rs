//! Source invariants for two guards that no runtime test can observe:
//!
//! 1. Every redirect-safe `reqwest::Client` builder in the workspace fails closed
//!    when `build()` fails. Falling back to `Client::default()` (via
//!    `unwrap_or_default()` or similar) would silently return a client that follows
//!    redirects, which is the behaviour `Policy::none()` exists to prevent. A builder
//!    failure does not happen in tests, so only the source shape can be pinned.
//! 2. Both keystore copies compare the keystore MAC in constant time (`ct_eq`). A
//!    plain `==` behaves identically in every test; only its timing differs.
//!
//! Each scanner has a negative control on the previous shape.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Builder chains (from `Client::builder()` to the end of the statement) that set a
/// redirect policy and then fall back to a default client when `build()` fails.
fn fail_open_redirect_builders(src: &str) -> Vec<String> {
    let code = strip_line_comments(src);
    let mut bad = Vec::new();
    let mut rest = code.as_str();
    while let Some(i) = rest.find("Client::builder()") {
        let chain_start = &rest[i..];
        let end = chain_start.find(';').unwrap_or(chain_start.len());
        let chain = &chain_start[..end];
        let compact: String = chain.split_whitespace().collect();
        if compact.contains(".redirect(") {
            let tail = compact.split(".build()").nth(1).unwrap_or("");
            let fails_open = tail.starts_with(".unwrap_or_default()")
                || tail.starts_with(".unwrap_or_else(")
                || tail.starts_with(".unwrap_or(")
                || tail.starts_with(".ok()");
            if fails_open {
                bad.push(compact.chars().take(160).collect());
            }
        }
        rest = &chain_start[end..];
    }
    bad
}

const CRATES: [&str; 3] = [
    "pool-coordinator/src",
    "training-worker/src",
    "training-coordinator/src",
];

#[test]
fn redirect_safe_clients_fail_closed_when_the_builder_fails() {
    let root = workspace_root();
    let mut files = Vec::new();
    for c in CRATES {
        rust_files(&root.join(c), &mut files);
    }
    assert!(
        files.len() > 10,
        "expected to scan the workspace sources, found {}",
        files.len()
    );
    let mut builders = 0usize;
    let mut offenders = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f).expect("read source");
        builders += src.matches("Client::builder()").count();
        for b in fail_open_redirect_builders(&src) {
            offenders.push(format!(
                "{}: {b}",
                f.strip_prefix(&root).unwrap_or(f).display()
            ));
        }
    }
    assert!(
        builders >= 2,
        "expected reqwest builders in the workspace, found {builders}"
    );
    assert!(
        offenders.is_empty(),
        "redirect-safe client builders that fall back to a default client:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn redirect_builder_scanner_negative_control() {
    let old = r#"
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default();
        let ok = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("redirect-safe client");
        let plain = reqwest::Client::builder().build().unwrap_or_default();
    "#;
    assert_eq!(
        fail_open_redirect_builders(old).len(),
        1,
        "only the redirect builder that falls back must be flagged"
    );
}

/// Problems with the keystore MAC comparison in one wallet copy.
fn mac_compare_problems(src: &str) -> Vec<&'static str> {
    let code = strip_line_comments(src);
    let compact: String = code.split_whitespace().collect();
    let mut p = Vec::new();
    if !compact.contains("mac_actual.as_slice().ct_eq(mac_expected.as_slice())") {
        p.push("MAC is not compared with ct_eq");
    }
    for bad in [
        "mac_actual.as_slice()!=mac_expected",
        "mac_actual.as_slice()==mac_expected",
        "mac_actual[..]!=mac_expected",
        "mac_actual[..]==mac_expected",
        "mac_expected.as_slice()!=mac_actual",
        "mac_expected.as_slice()==mac_actual",
    ] {
        if compact.contains(bad) {
            p.push("MAC compared with a variable-time operator");
        }
    }
    p
}

#[test]
fn both_keystore_copies_compare_the_mac_in_constant_time() {
    let root = workspace_root();
    for copy in [
        "pool-coordinator/src/wallet.rs",
        "training-worker/src/wallet.rs",
    ] {
        let src = std::fs::read_to_string(root.join(copy)).expect("read wallet copy");
        let problems = mac_compare_problems(&src);
        assert!(problems.is_empty(), "{copy}: {problems:?}");
    }
}

#[test]
fn mac_scanner_negative_control() {
    let old = "let mac_actual = hasher.finalize();\nif mac_actual.as_slice() != mac_expected.as_slice() {\n    return Err(WalletError::InvalidPassphrase);\n}\n";
    assert_eq!(mac_compare_problems(old).len(), 2);
}
