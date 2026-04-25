//! Event topic-0 classifier + raw log type shared across the
//! training-worker HTTP chain clients (CM-07 training, CM-08
//! pipeline). Lets `main.rs` drive a unified event polling loop
//! that dispatches based on the decoded event name.
//!
//! # Why classify on topic0 only
//!
//! For S1 the daemon needs to know *what happened* and *to which
//! job*, not the full payload — the operator-visible value is a
//! log line like "EpochCommitted jobId=42 block=12345". Full
//! payload decode is the SDK's job (parseTrainingEvents) and
//! is available to operators via offline analysis. This module
//! is scoped to "recognize the event; reject noise".

use ethereum_types::{H160, H256, U256};
use sha3::{Digest, Keccak256};

/// Raw log envelope returned by `eth_getLogs`, reshaped for
/// consumers who want to dedupe or forward without decoding.
#[derive(Debug, Clone)]
pub struct RawLog {
    pub address: H160,
    pub topics: Vec<H256>,
    pub data: Vec<u8>,
    pub block_number: u64,
    pub tx_hash: H256,
    pub log_index: u32,
}

impl RawLog {
    /// (tx_hash, log_index) dedup key.
    pub fn dedup_key(&self) -> (H256, u32) {
        (self.tx_hash, self.log_index)
    }
}

/// All CM-07 + CM-08 event kinds we classify. Unknown topics are
/// reported as [`EventKind::Unknown`] so callers can count them
/// without ignoring them silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    // ── CM-07 DataParallel training (ComputePoolTraining) ──
    TrainingJobOpened,
    WorkerJoined,
    RecruitmentClosed,
    CoordinatorReassigned,
    EpochCommitted,
    EpochPaymentReleased,
    ChallengeOpened,
    ChallengeVoted,
    ChallengeResolved,
    WorkerStakeReturned,
    TrainingJobCompleted,
    TrainingJobAborted,
    // ── CM-08 PipelineParallel inference (ComputePoolPipeline) ──
    PipelineJobCreated,
    StageAssigned,
    PipelineJobActivated,
    PipelineRequestSubmitted,
    StageServed,
    PipelineRequestCompleted,
    PipelineRequestFailed,
    StageFaulted,
    StageReassigned,
    PipelineJobDraining,
    PipelineJobTerminated,
    // ── Fallback ──
    Unknown,
}

impl EventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventKind::TrainingJobOpened => "TrainingJobOpened",
            EventKind::WorkerJoined => "WorkerJoined",
            EventKind::RecruitmentClosed => "RecruitmentClosed",
            EventKind::CoordinatorReassigned => "CoordinatorReassigned",
            EventKind::EpochCommitted => "EpochCommitted",
            EventKind::EpochPaymentReleased => "EpochPaymentReleased",
            EventKind::ChallengeOpened => "ChallengeOpened",
            EventKind::ChallengeVoted => "ChallengeVoted",
            EventKind::ChallengeResolved => "ChallengeResolved",
            EventKind::WorkerStakeReturned => "WorkerStakeReturned",
            EventKind::TrainingJobCompleted => "TrainingJobCompleted",
            EventKind::TrainingJobAborted => "TrainingJobAborted",
            EventKind::PipelineJobCreated => "PipelineJobCreated",
            EventKind::StageAssigned => "StageAssigned",
            EventKind::PipelineJobActivated => "JobActivated",
            EventKind::PipelineRequestSubmitted => "PipelineRequestSubmitted",
            EventKind::StageServed => "StageServed",
            EventKind::PipelineRequestCompleted => "PipelineRequestCompleted",
            EventKind::PipelineRequestFailed => "PipelineRequestFailed",
            EventKind::StageFaulted => "StageFaulted",
            EventKind::StageReassigned => "StageReassigned",
            EventKind::PipelineJobDraining => "JobDraining",
            EventKind::PipelineJobTerminated => "JobTerminated",
            EventKind::Unknown => "Unknown",
        }
    }
}

/// Decode a JSON log entry from `eth_getLogs` into a [`RawLog`].
/// Shared between `http_chain_training.rs` and
/// `http_chain_pipeline.rs` so both event-poll paths share the
/// same parser.
///
/// Returns `None` if the entry is malformed (missing topics,
/// unexpected sizes). Caller logs + skips; a bad individual log
/// shouldn't fail the whole poll tick.
pub fn decode_log_entry(entry: &serde_json::Value) -> Option<RawLog> {
    let topics_raw = entry.get("topics")?.as_array()?;
    let mut topics = Vec::with_capacity(topics_raw.len());
    for t in topics_raw {
        let s = t.as_str()?;
        let bytes = hex::decode(s.trim_start_matches("0x")).ok()?;
        if bytes.len() != 32 {
            return None;
        }
        topics.push(H256::from_slice(&bytes));
    }
    let address = entry
        .get("address")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s.trim_start_matches("0x")).ok())
        .and_then(|bytes| {
            if bytes.len() == 20 {
                Some(H160::from_slice(&bytes))
            } else {
                None
            }
        })
        .unwrap_or_default();
    let data = entry
        .get("data")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s.trim_start_matches("0x")).ok())
        .unwrap_or_default();
    let block_number = entry
        .get("blockNumber")
        .and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    let tx_hash = entry
        .get("transactionHash")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s.trim_start_matches("0x")).ok())
        .and_then(|bytes| {
            if bytes.len() == 32 {
                Some(H256::from_slice(&bytes))
            } else {
                None
            }
        })
        .unwrap_or_default();
    let log_index = entry
        .get("logIndex")
        .and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0) as u32;
    Some(RawLog {
        address,
        topics,
        data,
        block_number,
        tx_hash,
        log_index,
    })
}

/// Build the topic-filter JSON for `eth_getLogs` that filters by
/// indexed jobId. All CM-07 and CM-08 events use jobId as their
/// first indexed param, so this is a one-shot filter.
pub fn job_id_topic_filter(job_id: u64) -> serde_json::Value {
    let mut buf = [0u8; 32];
    U256::from(job_id).to_big_endian(&mut buf);
    serde_json::json!([
        serde_json::Value::Null,
        format!("0x{}", hex::encode(buf))
    ])
}

fn keccak(sig: &str) -> H256 {
    let mut h = Keccak256::new();
    h.update(sig.as_bytes());
    let out = h.finalize();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    H256::from(buf)
}

/// Classify a log by its topic[0]. Returns Unknown if we don't
/// recognize the signature — callers log + count for visibility.
pub fn classify(topic0: H256) -> EventKind {
    // Precomputed lazy_static could optimize this; the daemon polls
    // at 3-second cadence so the linear scan is <1µs per event.
    let candidates: &[(EventKind, &str)] = &[
        // CM-07 signatures (exact match with ComputePoolTraining.sol events)
        (EventKind::TrainingJobOpened, "TrainingJobOpened(uint256,address,bytes32,bytes32,uint32,uint32)"),
        (EventKind::WorkerJoined, "WorkerJoined(uint256,address,uint128)"),
        (EventKind::RecruitmentClosed, "RecruitmentClosed(uint256,uint32,address)"),
        (EventKind::CoordinatorReassigned, "CoordinatorReassigned(uint256,address,address,uint128)"),
        (EventKind::EpochCommitted, "EpochCommitted(uint256,uint32,bytes32)"),
        (EventKind::EpochPaymentReleased, "EpochPaymentReleased(uint256,uint32,address,uint128)"),
        (EventKind::ChallengeOpened, "ChallengeOpened(uint256,uint32,uint32,address,address,uint128)"),
        (EventKind::ChallengeVoted, "ChallengeVoted(uint256,uint32,uint32,address,address,bool)"),
        (EventKind::ChallengeResolved, "ChallengeResolved(uint256,uint32,uint32,address,bool,uint128)"),
        (EventKind::WorkerStakeReturned, "WorkerStakeReturned(uint256,address,uint128)"),
        (EventKind::TrainingJobCompleted, "TrainingJobCompleted(uint256,bytes32)"),
        (EventKind::TrainingJobAborted, "TrainingJobAborted(uint256,string)"),
        // CM-08 signatures (exact match with ComputePoolPipeline.sol events)
        (EventKind::PipelineJobCreated, "PipelineJobCreated(uint256,address,uint32,bytes32)"),
        (EventKind::StageAssigned, "StageAssigned(uint256,uint32,address,uint128)"),
        (EventKind::PipelineJobActivated, "JobActivated(uint256)"),
        (EventKind::PipelineRequestSubmitted, "PipelineRequestSubmitted(uint256,uint256,address,uint128)"),
        (EventKind::StageServed, "StageServed(uint256,uint32,address,uint128)"),
        (EventKind::PipelineRequestCompleted, "PipelineRequestCompleted(uint256)"),
        (EventKind::PipelineRequestFailed, "PipelineRequestFailed(uint256,uint128)"),
        (EventKind::StageFaulted, "StageFaulted(uint256,uint32,address)"),
        (EventKind::StageReassigned, "StageReassigned(uint256,uint32,address,address)"),
        (EventKind::PipelineJobDraining, "JobDraining(uint256)"),
        (EventKind::PipelineJobTerminated, "JobTerminated(uint256)"),
    ];

    for (kind, sig) in candidates {
        if keccak(sig) == topic0 {
            return *kind;
        }
    }
    EventKind::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_training_signatures() {
        assert_eq!(
            classify(keccak(
                "TrainingJobOpened(uint256,address,bytes32,bytes32,uint32,uint32)"
            )),
            EventKind::TrainingJobOpened
        );
        assert_eq!(
            classify(keccak("EpochCommitted(uint256,uint32,bytes32)")),
            EventKind::EpochCommitted
        );
        assert_eq!(
            classify(keccak("TrainingJobCompleted(uint256,bytes32)")),
            EventKind::TrainingJobCompleted
        );
    }

    #[test]
    fn classify_pipeline_signatures() {
        assert_eq!(
            classify(keccak(
                "PipelineRequestSubmitted(uint256,uint256,address,uint128)"
            )),
            EventKind::PipelineRequestSubmitted
        );
        assert_eq!(
            classify(keccak("StageReassigned(uint256,uint32,address,address)")),
            EventKind::StageReassigned
        );
    }

    #[test]
    fn unknown_topic_classifies_as_unknown() {
        let noise = H256::repeat_byte(0xFF);
        assert_eq!(classify(noise), EventKind::Unknown);
    }

    #[test]
    fn event_kind_strings_are_stable() {
        // These strings appear in operator-visible logs; stable
        // strings keep alert rules / grep patterns from breaking.
        assert_eq!(EventKind::TrainingJobOpened.as_str(), "TrainingJobOpened");
        assert_eq!(EventKind::StageFaulted.as_str(), "StageFaulted");
    }

    #[test]
    fn raw_log_dedup_key_is_tuple() {
        let log = RawLog {
            address: H160::repeat_byte(0xAA),
            topics: vec![],
            data: vec![],
            block_number: 42,
            tx_hash: H256::repeat_byte(0xBB),
            log_index: 7,
        };
        assert_eq!(log.dedup_key(), (H256::repeat_byte(0xBB), 7));
    }
}
