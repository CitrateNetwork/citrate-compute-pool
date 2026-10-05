//! Trainer input rows from a verified trajectory export (HUP-S9.3, RA-15).
//!
//! A member's device trains on its own verified, redacted Hermes trajectories: the JSONL that
//! citrate-agent-runtime's `export_verified` writes (one `{"messages": [...], "metadata": {...}}`
//! per verified turn). This module turns that export into the rows the operator's trainer reads
//! (`CITRATE_LORA_DATASET`, see [`super::trainer`]), in the format `citrate-fl-sft-v1`:
//!
//! ```json
//! {"messages": [...unchanged...],
//!  "tools": ["memory_search", ...the 29 parity-v1 tool names...],
//!  "metadata": {"model": "...", "workflow": null, "step": null, "verifiers": ["..."],
//!               "format": "citrate-fl-sft-v1", "tool_schema": "parity-v1",
//!               "source": "hermes-verified", "split": "train", "example_id": "<sha256>"}}
//! ```
//!
//! What it guarantees, each covered by `dataset_tests.rs`:
//!
//! - **Redaction is preserved.** Messages are copied verbatim: the converter never rewrites,
//!   re-renders or un-escapes content, so every `[REDACTED:...]` and `[root:N]` marker the
//!   exporter wrote reaches the trainer unchanged. Redaction itself is the exporter's job.
//! - **No held-out item leaks.** An example with any user message whose normalized text hashes
//!   to an item of the held-out manifest (by default the embedded toolcall-v2 manifest, which
//!   citrate-core generates from its eval set) is dropped. The manifest carries hashes only.
//! - **Only the shipping tool surface is trained.** An example that calls a tool outside the
//!   29-tool parity-v1 schema (an MCP server's tool, a retired name) is dropped, as is one whose
//!   tool-call arguments are not a JSON object, or whose tool results do not answer a call.
//! - **No duplicates.** Rows are keyed by the sha256 of their messages; a repeat is dropped.
//! - Rows still pass [`super::runner::verify_dataset`], so the device runner accepts them.
//!
//! The input must be a verified export: a line that does not parse, carries a system prompt or
//! names no passing verifier is an error, not a skipped line, because such a file did not come
//! from `export_verified`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{sha256, B32};

/// The row format this module writes.
pub const ROW_FORMAT: &str = "citrate-fl-sft-v1";
/// The tool schema rows are trained against.
pub const TOOL_SCHEMA: &str = "parity-v1";
/// The source tag of rows made from a member's own verified trajectories.
pub const SOURCE_HERMES: &str = "hermes-verified";

/// The 29 tools of citrate-core `src/agent/parity/parity-v1.json` (`tools[].name`, in order):
/// the core-hosted tool surface the shipping agent offers.
pub const PARITY_V1_TOOLS: [&str; 29] = [
    "memory_search",
    "memory_recall",
    "app_navigate",
    "memory_assert",
    "journal_append",
    "journal_read",
    "node_status",
    "staking_status",
    "groups_list",
    "group_roster",
    "group_create",
    "group_invite",
    "directory_find",
    "skills_list",
    "skill_write",
    "skill_run",
    "widget_create",
    "models_list",
    "contract_deploy",
    "get_verified_source",
    "contract_view",
    "fl_round_plan",
    "fl_round_start",
    "belnap_codec",
    "gsheets_read",
    "gsheets_append",
    "schedule_list",
    "schedule_add",
    "calendar_list",
];

/// citrate-core `src/agent/eval/toolcall-v2.heldout.json`, copied byte for byte.
pub const TOOLCALL_V2_HELDOUT: &str = include_str!("../../fl-data/toolcall-v2.heldout.json");
/// Its sha256. citrate-core's `heldout.test.ts` regenerates the file from the eval set; a new
/// copy here must come with a new pin.
pub const TOOLCALL_V2_HELDOUT_SHA256: &str =
    "bd35360bc8f9549b0519c3a5f038c6234bc8feb69176c48daacfbe7c46047772";
/// The manifest format this module reads.
pub const HELDOUT_FORMAT: &str = "citrate-heldout-v1";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DatasetError {
    #[error("the export is not UTF-8")]
    NotUtf8,
    #[error("line {line}: not a verified trajectory export line: {detail}")]
    BadLine { line: usize, detail: String },
    #[error("held-out manifest: {0}")]
    Manifest(String),
    #[error("could not serialize a row: {0}")]
    Serialize(String),
}

/// The shared prompt normalization, specified identically in citrate-core
/// `src/agent/eval/heldout.ts`: lowercase; every character that is neither alphabetic nor
/// numeric becomes a space; whitespace runs collapse to one space; trimmed.
pub fn normalize_prompt(s: &str) -> String {
    let lowered = s.to_lowercase();
    let mapped: String = lowered
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    mapped.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    format: String,
    dataset: String,
    #[allow(dead_code)]
    normalization: String,
    count: usize,
    items: Vec<ManifestItem>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestItem {
    id: String,
    sha256: String,
}

/// The prompts a training set must not contain, as hashes of their normalized text.
#[derive(Debug, Clone)]
pub struct HeldOut {
    dataset: String,
    hashes: BTreeSet<B32>,
}

impl HeldOut {
    /// Parse a `citrate-heldout-v1` manifest.
    pub fn parse(json: &str) -> Result<Self, DatasetError> {
        let m: ManifestFile =
            serde_json::from_str(json).map_err(|e| DatasetError::Manifest(e.to_string()))?;
        if m.format != HELDOUT_FORMAT {
            return Err(DatasetError::Manifest(format!(
                "format {:?}, expected {HELDOUT_FORMAT:?}",
                m.format
            )));
        }
        if m.count != m.items.len() || m.items.is_empty() {
            return Err(DatasetError::Manifest(format!(
                "count {} does not match its {} items",
                m.count,
                m.items.len()
            )));
        }
        let mut hashes = BTreeSet::new();
        for it in &m.items {
            let h = super::parse_hex::<32>(&it.sha256)
                .map_err(|e| DatasetError::Manifest(format!("item {}: {e}", it.id)))?;
            hashes.insert(h);
        }
        Ok(Self {
            dataset: m.dataset,
            hashes,
        })
    }

    /// The embedded toolcall-v2 manifest.
    pub fn toolcall_v2() -> Result<Self, DatasetError> {
        Self::parse(TOOLCALL_V2_HELDOUT)
    }

    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Whether `text` normalizes to a held-out prompt.
    pub fn contains_prompt(&self, text: &str) -> bool {
        self.hashes
            .contains(&sha256(normalize_prompt(text).as_bytes()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: Function,
}

/// One message, exactly as the exporter wrote it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportMeta {
    model: String,
    workflow: Option<String>,
    step: Option<String>,
    verifiers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportLine {
    messages: Vec<Message>,
    metadata: ExportMeta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowMeta {
    pub model: String,
    pub workflow: Option<String>,
    pub step: Option<String>,
    pub verifiers: Vec<String>,
    pub format: String,
    pub tool_schema: String,
    pub source: String,
    pub split: String,
    pub example_id: String,
}

/// One trainer input row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainerRow {
    pub messages: Vec<Message>,
    pub tools: Vec<String>,
    pub metadata: RowMeta,
}

/// Why examples were left out. Counts only, never content.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Excluded {
    /// A user message matched the held-out manifest.
    pub held_out: u64,
    /// A tool call named a tool outside parity-v1.
    pub off_schema: u64,
    /// A tool call's arguments were not a JSON object.
    pub bad_arguments: u64,
    /// A tool result answered no call, or a call got no result.
    pub unpaired_tool_result: u64,
    /// The same messages as an earlier row.
    pub duplicate: u64,
}

/// What a conversion did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ConvertReport {
    pub format: String,
    pub held_out_dataset: String,
    pub held_out_items: u64,
    pub read: u64,
    pub written: u64,
    pub excluded: Excluded,
    /// Tool calls per tool across the written rows.
    pub tool_calls: BTreeMap<String, u64>,
}

/// The converted rows (JSONL) and the report.
#[derive(Debug, Clone)]
pub struct Converted {
    pub rows: Vec<TrainerRow>,
    pub report: ConvertReport,
}

impl Converted {
    /// One JSON object per line.
    pub fn to_jsonl(&self) -> Result<String, DatasetError> {
        let mut out = String::new();
        for r in &self.rows {
            out.push_str(
                &serde_json::to_string(r).map_err(|e| DatasetError::Serialize(e.to_string()))?,
            );
            out.push('\n');
        }
        Ok(out)
    }
}

enum Verdict {
    Keep,
    HeldOut,
    OffSchema,
    BadArguments,
    Unpaired,
}

fn judge(messages: &[Message], held: &HeldOut) -> Verdict {
    if messages
        .iter()
        .any(|m| m.role == "user" && held.contains_prompt(&m.content))
    {
        return Verdict::HeldOut;
    }
    let calls = || messages.iter().flat_map(|m| m.tool_calls.iter());
    if calls().any(|c| !PARITY_V1_TOOLS.contains(&c.function.name.as_str())) {
        return Verdict::OffSchema;
    }
    if calls().any(|c| {
        !matches!(
            serde_json::from_str::<serde_json::Value>(&c.function.arguments),
            Ok(serde_json::Value::Object(_))
        )
    }) {
        return Verdict::BadArguments;
    }
    let call_ids: BTreeSet<&str> = calls().map(|c| c.id.as_str()).collect();
    let answered: BTreeSet<&str> = messages
        .iter()
        .filter(|m| m.role == "tool")
        .filter_map(|m| m.tool_call_id.as_deref())
        .collect();
    let tool_msgs = messages.iter().filter(|m| m.role == "tool").count();
    if answered != call_ids || tool_msgs != call_ids.len() {
        return Verdict::Unpaired;
    }
    Verdict::Keep
}

fn check_line(line_no: usize, ex: &ExportLine) -> Result<(), DatasetError> {
    let bad = |detail: String| DatasetError::BadLine {
        line: line_no,
        detail,
    };
    if ex.messages.is_empty() {
        return Err(bad("no messages".into()));
    }
    if let Some(m) = ex
        .messages
        .iter()
        .find(|m| !matches!(m.role.as_str(), "user" | "assistant" | "tool"))
    {
        return Err(bad(format!(
            "role {:?} (the export never carries a system prompt)",
            m.role
        )));
    }
    if ex.messages[0].role != "user" {
        return Err(bad(
            "an exported turn starts with the member's message".into()
        ));
    }
    if ex.metadata.verifiers.is_empty() {
        return Err(bad("no verifier passed this turn".into()));
    }
    Ok(())
}

/// Convert a verified trajectory export into trainer rows.
pub fn convert_export(export: &[u8], held: &HeldOut) -> Result<Converted, DatasetError> {
    let text = std::str::from_utf8(export).map_err(|_| DatasetError::NotUtf8)?;
    let mut report = ConvertReport {
        format: ROW_FORMAT.into(),
        held_out_dataset: held.dataset().to_string(),
        held_out_items: held.len() as u64,
        ..Default::default()
    };
    let tools: Vec<String> = PARITY_V1_TOOLS.iter().map(|t| (*t).to_string()).collect();
    let mut seen: BTreeSet<B32> = BTreeSet::new();
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let ex: ExportLine = serde_json::from_str(line).map_err(|e| DatasetError::BadLine {
            line: i + 1,
            detail: e.to_string(),
        })?;
        check_line(i + 1, &ex)?;
        report.read += 1;
        match judge(&ex.messages, held) {
            Verdict::Keep => {}
            Verdict::HeldOut => {
                report.excluded.held_out += 1;
                continue;
            }
            Verdict::OffSchema => {
                report.excluded.off_schema += 1;
                continue;
            }
            Verdict::BadArguments => {
                report.excluded.bad_arguments += 1;
                continue;
            }
            Verdict::Unpaired => {
                report.excluded.unpaired_tool_result += 1;
                continue;
            }
        }
        let canonical = serde_json::to_string(&ex.messages)
            .map_err(|e| DatasetError::Serialize(e.to_string()))?;
        let id = sha256(canonical.as_bytes());
        if !seen.insert(id) {
            report.excluded.duplicate += 1;
            continue;
        }
        for c in ex.messages.iter().flat_map(|m| m.tool_calls.iter()) {
            *report
                .tool_calls
                .entry(c.function.name.clone())
                .or_default() += 1;
        }
        rows.push(TrainerRow {
            messages: ex.messages,
            tools: tools.clone(),
            metadata: RowMeta {
                model: ex.metadata.model,
                workflow: ex.metadata.workflow,
                step: ex.metadata.step,
                verifiers: ex.metadata.verifiers,
                format: ROW_FORMAT.into(),
                tool_schema: TOOL_SCHEMA.into(),
                source: SOURCE_HERMES.into(),
                split: "train".into(),
                example_id: hex::encode(id),
            },
        });
        report.written += 1;
    }
    Ok(Converted { rows, report })
}

#[cfg(test)]
mod tests {
    include!("dataset_tests.rs");
}
