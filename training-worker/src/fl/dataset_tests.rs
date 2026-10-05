// HUP-S9.3 / RA-15: trajectory export -> trainer rows. The fixture is the real output of
// citrate-agent-runtime `export_verified` (agent-trajectory/tests/fl_fixture_tests.rs), copied
// byte for byte; both repos pin its sha256.
use super::*;

const EXPORT_FIXTURE: &str = include_str!("../../tests/fixtures/fl/export-v1.jsonl");
/// Pinned in citrate-agent-runtime `agent-trajectory/tests/fl_fixture_tests.rs` too.
const FL_EXPORT_V1_SHA256: &str =
    "3db35d06ea8ff9bdcfabf39abcb16e73b9a43eccdfa8747c5db4ebb82273a719";
/// Pinned in citrate-core `src/agent/eval/heldout.test.ts` too.
const PARITY_V1_TOOL_NAMES_SHA256: &str =
    "15534eb4d24830b58acb575a01cf6e7070e7ffb9ac81a463e8fe69350b3eecee";

fn held() -> HeldOut {
    HeldOut::toolcall_v2().expect("embedded manifest")
}

fn convert_fixture() -> Converted {
    convert_export(EXPORT_FIXTURE.as_bytes(), &held()).expect("convert")
}

#[test]
fn the_pinned_files_are_the_ones_the_other_repos_pin() {
    assert_eq!(
        hex::encode(sha256(EXPORT_FIXTURE.as_bytes())),
        FL_EXPORT_V1_SHA256
    );
    assert_eq!(
        hex::encode(sha256(TOOLCALL_V2_HELDOUT.as_bytes())),
        TOOLCALL_V2_HELDOUT_SHA256
    );
    assert_eq!(
        hex::encode(sha256(PARITY_V1_TOOLS.join("\n").as_bytes())),
        PARITY_V1_TOOL_NAMES_SHA256
    );
}

#[test]
fn the_toolcall_v2_manifest_holds_every_eval_item() {
    let h = held();
    assert_eq!(h.dataset(), "toolcall-v2");
    assert_eq!(h.len(), 79);
    // tc-node-height, through the normalization in any casing or punctuation.
    assert!(h.contains_prompt("What block height is my node at right now?"));
    assert!(h.contains_prompt("  WHAT block-height is my node at... right now"));
    assert!(!h.contains_prompt("What block height was my node at yesterday?"));
}

/// The same vectors as citrate-core `heldout.test.ts`.
#[test]
fn normalization_matches_the_core_vectors() {
    for (input, want) in [
        (
            "What block height is my node at right now?",
            "what block height is my node at right now",
        ),
        (
            "what block height is my node at, right now",
            "what block height is my node at right now",
        ),
        ("  Héllo,\tWORLD!! 42 ", "héllo world 42"),
        ("tab\nnew-line_under", "tab new line under"),
        ("0x1111 → [REDACTED:email]", "0x1111 redacted email"),
        ("", ""),
    ] {
        assert_eq!(normalize_prompt(input), want, "{input:?}");
    }
}

#[test]
fn the_fixture_converts_with_each_exclusion_counted() {
    let c = convert_fixture();
    let r = &c.report;
    assert_eq!(r.read, 6);
    assert_eq!(r.written, 2);
    assert_eq!(
        r.excluded,
        Excluded {
            held_out: 1,
            off_schema: 1,
            bad_arguments: 1,
            unpaired_tool_result: 0,
            duplicate: 1,
        }
    );
    assert_eq!(r.held_out_dataset, "toolcall-v2");
    assert_eq!(r.held_out_items, 79);
    assert_eq!(r.tool_calls.get("node_status"), Some(&1));
    assert_eq!(r.tool_calls.get("journal_append"), Some(&1));
    assert_eq!(r.tool_calls.len(), 2);
    for row in &c.rows {
        assert_eq!(row.metadata.format, ROW_FORMAT);
        assert_eq!(row.metadata.tool_schema, "parity-v1");
        assert_eq!(row.metadata.source, "hermes-verified");
        assert_eq!(row.metadata.split, "train");
        assert_eq!(row.tools.len(), 29);
    }
}

#[test]
fn no_held_out_item_reaches_the_trainer() {
    let c = convert_fixture();
    let h = held();
    for row in &c.rows {
        for m in row.messages.iter().filter(|m| m.role == "user") {
            assert!(!h.contains_prompt(&m.content), "{:?} leaked", m.content);
        }
    }
    let jsonl = c.to_jsonl().expect("jsonl");
    assert!(!jsonl.to_lowercase().contains("what block height"));
}

#[test]
fn redaction_is_preserved_byte_for_byte() {
    let c = convert_fixture();
    // Every written row's messages are exactly the export line's messages.
    let originals: Vec<serde_json::Value> = EXPORT_FIXTURE
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("line")["messages"].clone())
        .collect();
    for row in &c.rows {
        let mine = serde_json::to_value(&row.messages).expect("value");
        assert!(originals.contains(&mine), "row messages were rewritten");
    }
    let jsonl = c.to_jsonl().expect("jsonl");
    for marker in [
        "[REDACTED:email]",
        "[REDACTED:address]",
        "[REDACTED:path]",
        "[root:0]/README.md",
    ] {
        assert!(jsonl.contains(marker), "{marker} lost");
    }
    for raw in ["member@example.org", "0x1111111111111111111111111111111111111111", "/Users/"] {
        assert!(!jsonl.contains(raw), "{raw} appeared");
    }
}

#[test]
fn rows_are_accepted_by_the_device_runner() {
    let jsonl = convert_fixture().to_jsonl().expect("jsonl");
    assert_eq!(
        crate::fl::runner::verify_dataset(jsonl.as_bytes()).expect("verify"),
        2
    );
}

#[test]
fn example_ids_are_the_sha256_of_the_messages() {
    for row in convert_fixture().rows {
        let canonical = serde_json::to_string(&row.messages).expect("json");
        assert_eq!(row.metadata.example_id, hex::encode(sha256(canonical.as_bytes())));
    }
}

fn line(messages: serde_json::Value, verifiers: serde_json::Value) -> String {
    serde_json::json!({
        "messages": messages,
        "metadata": {"model": "m", "workflow": null, "step": null, "verifiers": verifiers}
    })
    .to_string()
}

#[test]
fn a_line_with_a_system_prompt_or_no_verifier_is_an_error_not_a_skip() {
    let sys = line(
        serde_json::json!([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]),
        serde_json::json!(["v"]),
    );
    assert!(matches!(
        convert_export(sys.as_bytes(), &held()),
        Err(DatasetError::BadLine { line: 1, .. })
    ));
    let unverified = line(
        serde_json::json!([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]),
        serde_json::json!([]),
    );
    assert!(matches!(
        convert_export(unverified.as_bytes(), &held()),
        Err(DatasetError::BadLine { line: 1, .. })
    ));
    let starts_with_answer = line(
        serde_json::json!([{"role": "assistant", "content": "a"}]),
        serde_json::json!(["v"]),
    );
    assert!(convert_export(starts_with_answer.as_bytes(), &held()).is_err());
    assert!(convert_export(b"not json\n", &held()).is_err());
    assert_eq!(
        convert_export(&[0xff, 0xfe], &held()).err(),
        Some(DatasetError::NotUtf8)
    );
}

#[test]
fn an_unknown_field_means_the_export_format_moved() {
    let mut v: serde_json::Value = serde_json::from_str(
        EXPORT_FIXTURE.lines().next().expect("first line"),
    )
    .expect("json");
    v["metadata"]["session_id"] = serde_json::json!("sess-1");
    assert!(convert_export(v.to_string().as_bytes(), &held()).is_err());
}

#[test]
fn a_tool_result_that_answers_no_call_is_dropped() {
    let orphan = line(
        serde_json::json!([
            {"role": "user", "content": "Is my node synced?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "node_status", "arguments": "{}"}}]},
            {"role": "tool", "content": "{}", "tool_call_id": "c9"},
            {"role": "assistant", "content": "yes"}
        ]),
        serde_json::json!(["v"]),
    );
    let c = convert_export(orphan.as_bytes(), &held()).expect("convert");
    assert_eq!(c.report.written, 0);
    assert_eq!(c.report.excluded.unpaired_tool_result, 1);
}

#[test]
fn a_plain_answer_without_tools_is_kept() {
    let plain = line(
        serde_json::json!([
            {"role": "user", "content": "Thanks, that is all for today."},
            {"role": "assistant", "content": "You're welcome."}
        ]),
        serde_json::json!(["answer_contains welcome"]),
    );
    let c = convert_export(plain.as_bytes(), &held()).expect("convert");
    assert_eq!(c.report.written, 1);
    assert!(c.report.tool_calls.is_empty());
}

#[test]
fn a_bad_manifest_is_refused() {
    assert!(HeldOut::parse("{}").is_err());
    assert!(HeldOut::parse(
        r#"{"format":"other","dataset":"d","normalization":"n","count":0,"items":[]}"#
    )
    .is_err());
    assert!(HeldOut::parse(
        r#"{"format":"citrate-heldout-v1","dataset":"d","normalization":"n","count":2,"items":[{"id":"a","sha256":"00"}]}"#
    )
    .is_err());
    assert!(HeldOut::parse(
        r#"{"format":"citrate-heldout-v1","dataset":"d","normalization":"n","count":1,"items":[{"id":"a","sha256":"zz"}]}"#
    )
    .is_err());
}
