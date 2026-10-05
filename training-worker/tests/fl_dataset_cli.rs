//! `citrate-fl-dataset` end to end on the exported fixture (HUP-S9.3, RA-15).
use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_citrate-fl-dataset")
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fl/export-v1.jsonl")
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fl-dataset-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("dir");
    d
}

#[test]
fn converts_the_fixture_and_never_overwrites() {
    let d = tmp("ok");
    let out = d.join("rows.jsonl");
    let report = d.join("report.json");
    let run = || {
        Command::new(bin())
            .arg("--export")
            .arg(fixture())
            .arg("--out")
            .arg(&out)
            .arg("--report")
            .arg(&report)
            .output()
            .expect("run")
    };
    let first = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let rows = std::fs::read_to_string(&out).expect("rows");
    assert_eq!(rows.lines().count(), 2);
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).expect("report")).expect("json");
    assert_eq!(r["written"], 2);
    assert_eq!(r["excluded"]["held_out"], 1);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&out).expect("meta").permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    // A second run refuses to replace either file.
    let again = run();
    assert_eq!(again.status.code(), Some(1));
    assert_eq!(std::fs::read_to_string(&out).expect("rows"), rows);
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn nothing_surviving_is_exit_two_and_writes_no_rows() {
    let d = tmp("empty");
    let export = d.join("export.jsonl");
    // Only the held-out item survives the exporter; the converter must drop it.
    let held_out_line = std::fs::read_to_string(fixture())
        .expect("fixture")
        .lines()
        .nth(3)
        .expect("line 4")
        .to_string();
    std::fs::write(&export, format!("{held_out_line}\n")).expect("write");
    let out = d.join("rows.jsonl");
    let o = Command::new(bin())
        .arg("--export")
        .arg(&export)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run");
    assert_eq!(o.status.code(), Some(2));
    assert!(!out.exists());
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn unknown_options_are_refused() {
    let o = Command::new(bin())
        .args(["--export", "x", "--upload", "y"])
        .output()
        .expect("run");
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown option --upload"));
}
