//! Aggregating a federated LoRA round (HUP-S9.2, FL_ROUND_V1 §5–§7).
//!
//! The aggregator reads the coordinator's state for the round's finished
//! `lora_delta` jobs, verifies every contribution from first principles (the
//! signature recovers to the worker that the coordinator recorded, the worker
//! is on the roster, the artifact hashes to what was signed and re-derives the
//! signed delta root, its header names this round and start adapter), and
//! refuses the round outright when fewer than the minimum remain.
//!
//! It then builds each chunk's `0x0110` input from the participants' rows in
//! canonical order (ascending worker address) and has a **Citrate node** run
//! the precompile, by `eth_call` to `FederatedRoundLedger.belnapAggregate`. No
//! aggregate is computed in this crate. The outputs are folded into the merged
//! adapter (start + aggregate), and the round is summarised as a bundle plus an
//! **unsigned** `commitRound` intent: signing and sending it is the
//! coordinator operator's ceremony, never this tool's (Rule 3).

use std::collections::BTreeMap;
use std::path::Path;

use citrate_training_worker::fl::belnap::{chunk_input, parse_output};
use citrate_training_worker::fl::delta::{self, Artifact};
use citrate_training_worker::fl::gguf::Adapter;
use citrate_training_worker::fl::round::{LoraDeltaPayload, LoraDeltaResult, RoundConfig};
use citrate_training_worker::fl::{hex0x, hexser, keccak, sha256, tree, Addr, B32};
use citrate_training_worker::wallet::Wallet;
use serde::{Deserialize, Serialize};

use crate::job::JobId;
use crate::state::{JobStatus, State};

pub const BUNDLE_VERSION: &str = "citrate-fl-round-bundle/1";

#[derive(Debug, thiserror::Error)]
pub enum RoundError {
    #[error("round config: {0}")]
    Config(String),
    #[error("only {got} valid contributions; the round needs {need}")]
    TooFew { got: usize, need: u16 },
    #[error("start adapter: {0}")]
    Start(String),
    #[error("node: {0}")]
    Node(String),
    #[error("chunk {chunk}: {msg}")]
    Chunk { chunk: usize, msg: String },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

/// Runs `0x0110` somewhere that has it.
#[async_trait::async_trait]
pub trait BelnapBackend: Send + Sync {
    async fn aggregate(&self, input: &[u8]) -> Result<Vec<u8>, RoundError>;
}

/// `eth_call` to `FederatedRoundLedger.belnapAggregate(bytes)` on a Citrate
/// node. Data source: the node's `0x0110` precompile, reached through the
/// ledger's view (a direct `eth_call` to a precompile address is not served by
/// every node build).
pub struct RpcBelnap {
    rpc: String,
    ledger: Addr,
    http: reqwest::Client,
}

impl RpcBelnap {
    /// The RPC URL must be https, or http on loopback (CP-B-006 outbound
    /// policy, shared with the worker).
    pub fn new(rpc: impl Into<String>, ledger: Addr) -> Result<Self, RoundError> {
        let rpc = rpc.into();
        citrate_training_worker::outbound::validate_outbound_url(&rpc)
            .map_err(|e| RoundError::Node(e.to_string()))?;
        Ok(Self {
            rpc,
            ledger,
            http: citrate_training_worker::outbound::redirect_safe_client(
                std::time::Duration::from_secs(60),
            ),
        })
    }
}

/// `keccak256(signature)[..4]`.
pub fn selector(sig: &str) -> [u8; 4] {
    let h = keccak(&[sig.as_bytes()]);
    [h[0], h[1], h[2], h[3]]
}

/// ABI-encode `f(bytes)`.
pub fn encode_bytes_call(sig: &str, data: &[u8]) -> Vec<u8> {
    let mut out = selector(sig).to_vec();
    let mut w = [0u8; 32];
    w[31] = 0x20;
    out.extend_from_slice(&w);
    let mut len = [0u8; 32];
    len[24..].copy_from_slice(&(data.len() as u64).to_be_bytes());
    out.extend_from_slice(&len);
    out.extend_from_slice(data);
    out.resize(4 + 64 + data.len().div_ceil(32) * 32, 0);
    out
}

/// Decode a single `bytes` return value.
pub fn decode_bytes_return(ret: &[u8]) -> Result<Vec<u8>, String> {
    let word = |i: usize| -> Result<u64, String> {
        let w = ret
            .get(i..i + 32)
            .ok_or_else(|| "return data too short".to_string())?;
        if w[..24].iter().any(|b| *b != 0) {
            return Err("ABI word out of range".into());
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&w[24..]);
        Ok(u64::from_be_bytes(b))
    };
    let off = usize::try_from(word(0)?).map_err(|e| e.to_string())?;
    let len = usize::try_from(word(off)?).map_err(|e| e.to_string())?;
    let start = off + 32;
    ret.get(start..start + len)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| "return data shorter than its length".into())
}

#[async_trait::async_trait]
impl BelnapBackend for RpcBelnap {
    async fn aggregate(&self, input: &[u8]) -> Result<Vec<u8>, RoundError> {
        let data = encode_bytes_call("belnapAggregate(bytes)", input);
        let body = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "eth_call",
            "params": [{"to": hex0x(&self.ledger), "data": hex0x(&data)}, "latest"],
        });
        let res: serde_json::Value = self
            .http
            .post(&self.rpc)
            .json(&body)
            .send()
            .await
            .map_err(|e| RoundError::Node(e.to_string()))?
            .json()
            .await
            .map_err(|e| RoundError::Node(e.to_string()))?;
        if let Some(e) = res.get("error") {
            return Err(RoundError::Node(format!("eth_call failed: {e}")));
        }
        let hexret = res
            .get("result")
            .and_then(|r| r.as_str())
            .ok_or_else(|| RoundError::Node("eth_call returned no result".into()))?;
        let raw = hex::decode(hexret.trim_start_matches("0x"))
            .map_err(|e| RoundError::Node(e.to_string()))?;
        decode_bytes_return(&raw).map_err(RoundError::Node)
    }
}

/// One verified contribution.
#[derive(Debug, Clone)]
pub struct Contribution {
    pub job: JobId,
    pub result: LoraDeltaResult,
    pub artifact: Artifact,
}

/// Why a submitted result was left out of the round.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Exclusion {
    pub job: String,
    pub reason: String,
}

/// Collect and verify the round's contributions from coordinator state and the
/// delta store. Every refusal is recorded, never silently dropped.
pub fn collect(
    state: &State,
    cfg: &RoundConfig,
    ordinal: u64,
    deltas: &Path,
    start: &Adapter,
) -> (Vec<Contribution>, Vec<Exclusion>) {
    let round_id = cfg.round_id(ordinal);
    let manifest = delta::manifest_hash(start);
    let expect_values = delta::value_count(start);
    let mut by_worker: BTreeMap<Addr, Contribution> = BTreeMap::new();
    let mut excluded = Vec::new();
    for (id, rec) in &state.jobs {
        let Ok(p) = serde_json::from_value::<LoraDeltaPayload>(rec.spec.payload.clone()) else {
            continue;
        };
        if p.round_id != round_id {
            continue;
        }
        let mut out = |reason: String| {
            excluded.push(Exclusion {
                job: id.0.clone(),
                reason,
            })
        };
        if p.config != *cfg {
            out("the job's round config differs from the round's".into());
            continue;
        }
        let JobStatus::Done { worker, .. } = rec.status else {
            continue;
        };
        let Some(raw) = &rec.result else {
            out("done without a result".into());
            continue;
        };
        let r: LoraDeltaResult = match serde_json::from_str(raw) {
            Ok(r) => r,
            Err(e) => {
                out(format!("result is not a lora_delta result: {e}"));
                continue;
            }
        };
        let submitter = worker.to_fixed_bytes();
        if r.worker != submitter {
            out("the result names a worker other than its submitter".into());
            continue;
        }
        if r.round_id != round_id {
            out("the result is for another round".into());
            continue;
        }
        if !cfg.in_roster(&r.worker) {
            out("the worker is not on the roster".into());
            continue;
        }
        let sig = match hex::decode(r.signature.trim_start_matches("0x")) {
            Ok(s) => s,
            Err(_) => {
                out("the delta signature is not hex".into());
                continue;
            }
        };
        match Wallet::recover_address(&r.digest(), &sig) {
            Ok(a) if a.to_fixed_bytes() == r.worker => {}
            _ => {
                out("the delta signature does not recover to the worker".into());
                continue;
            }
        }
        let path = deltas.join(format!("{}.fld", hex::encode(r.delta_sha256)));
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                out(format!("artifact {} unavailable: {e}", path.display()));
                continue;
            }
        };
        if sha256(&bytes) != r.delta_sha256 {
            out("the stored artifact does not hash to the signed value".into());
            continue;
        }
        let art = match Artifact::decode(&bytes, cfg.max_values) {
            Ok(a) => a,
            Err(e) => {
                out(format!("artifact: {e}"));
                continue;
            }
        };
        if art.round_id != round_id
            || art.worker != r.worker
            || art.start_adapter_sha256 != cfg.start_adapter_sha256
            || art.manifest_hash != manifest
            || art.chunk_dim != cfg.chunk_dim
            || art.value_scale_log2 != cfg.value_scale_log2
            || art.values.len() as u64 != expect_values
            || r.n_values != expect_values
            || r.chunk_dim != cfg.chunk_dim
        {
            out("the artifact header does not match the round".into());
            continue;
        }
        match delta::delta_root(&art.values, cfg.chunk_dim) {
            Ok(root) if root == r.delta_root => {}
            _ => {
                out("the artifact does not re-derive the signed delta root".into());
                continue;
            }
        }
        if by_worker.contains_key(&r.worker) {
            out("a second contribution from the same worker".into());
            continue;
        }
        by_worker.insert(
            r.worker,
            Contribution {
                job: id.clone(),
                result: r,
                artifact: art,
            },
        );
    }
    (by_worker.into_values().collect(), excluded)
}

/// One participant in the bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleParticipant {
    #[serde(with = "hexser")]
    pub worker: Addr,
    pub job: String,
    #[serde(with = "hexser")]
    pub delta_root: B32,
    #[serde(with = "hexser")]
    pub delta_sha256: B32,
    /// The worker's signed result, verbatim.
    pub result: LoraDeltaResult,
}

/// Everything needed to audit, replay and challenge a round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bundle {
    pub version: String,
    pub ordinal: u64,
    #[serde(with = "hexser")]
    pub round_id: B32,
    #[serde(with = "hexser")]
    pub config_hash: B32,
    pub config: RoundConfig,
    pub participants: Vec<BundleParticipant>,
    pub n_values: u64,
    pub chunks: u32,
    pub input_hashes: Vec<String>,
    pub output_hashes: Vec<String>,
    #[serde(with = "hexser")]
    pub participants_root: B32,
    #[serde(with = "hexser")]
    pub input_root: B32,
    #[serde(with = "hexser")]
    pub output_root: B32,
    #[serde(with = "hexser")]
    pub adapter_sha256: B32,
    /// The on-chain record digest (`FederatedRoundLedger.recordDigest`), which
    /// settlement anchors as the round's merged hash.
    #[serde(with = "hexser")]
    pub record_digest: B32,
    /// Per-coordinate Belnap state counts over the whole round:
    /// `[neither, true, false, both]`.
    pub state_counts: [u64; 4],
    pub excluded: Vec<Exclusion>,
}

/// The participant leaf payload: `keccak256(worker ‖ delta_root)`.
pub fn participant_payload(worker: &Addr, delta_root: &B32) -> B32 {
    keccak(&[worker, delta_root])
}

fn word_u64(v: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

fn word_addr(a: &Addr) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a);
    w
}

/// `keccak256(abi.encode(chainid, ledger, roundId, configHash, participantsRoot,
/// inputRoot, outputRoot, adapterHash, nValues, chunks, participants))`.
#[allow(clippy::too_many_arguments)]
pub fn record_digest(
    chain_id: u64,
    ledger: &Addr,
    round_id: &B32,
    config_hash: &B32,
    participants_root: &B32,
    input_root: &B32,
    output_root: &B32,
    adapter_hash: &B32,
    n_values: u64,
    chunks: u32,
    participants: u16,
) -> B32 {
    keccak(&[
        &word_u64(chain_id),
        &word_addr(ledger),
        round_id,
        config_hash,
        participants_root,
        input_root,
        output_root,
        adapter_hash,
        &word_u64(n_values),
        &word_u64(u64::from(chunks)),
        &word_u64(u64::from(participants)),
    ])
}

pub const COMMIT_SIG: &str =
    "commitRound((bytes32,uint64,bytes32,bytes32,uint16,uint64,bytes32,bytes32,uint32,bytes32))";

/// The unsigned `commitRound` calldata for a bundle.
pub fn commit_calldata(b: &Bundle) -> Vec<u8> {
    let mut out = selector(COMMIT_SIG).to_vec();
    out.extend_from_slice(&b.config.cluster_id);
    out.extend_from_slice(&word_u64(b.ordinal));
    out.extend_from_slice(&b.config_hash);
    out.extend_from_slice(&b.participants_root);
    out.extend_from_slice(&word_u64(b.participants.len() as u64));
    out.extend_from_slice(&word_u64(b.n_values));
    out.extend_from_slice(&b.input_root);
    out.extend_from_slice(&b.output_root);
    out.extend_from_slice(&word_u64(u64::from(b.chunks)));
    out.extend_from_slice(&b.adapter_sha256);
    out
}

/// An unsigned transaction for the operator's ceremony to review and sign.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intent {
    pub kind: String,
    pub chain_id: u64,
    pub to: String,
    pub data: String,
    pub summary: String,
}

pub fn commit_intent(b: &Bundle) -> Intent {
    Intent {
        kind: "flCommitRound".into(),
        chain_id: b.config.chain_id,
        to: hex0x(&b.config.ledger),
        data: hex0x(&commit_calldata(b)),
        summary: format!(
            "Commit federated round {} (ordinal {}) of cluster {}: {} participants, {} chunks, \
             merged adapter sha256 {}",
            hex0x(&b.round_id),
            b.ordinal,
            hex0x(&b.config.cluster_id),
            b.participants.len(),
            b.chunks,
            hex::encode(b.adapter_sha256)
        ),
    }
}

/// The aggregated round: its bundle and the merged adapter's bytes.
pub struct Outcome {
    pub bundle: Bundle,
    pub merged_adapter: Vec<u8>,
}

/// Aggregate verified contributions through `backend`.
pub async fn aggregate(
    cfg: &RoundConfig,
    ordinal: u64,
    contributions: Vec<Contribution>,
    excluded: Vec<Exclusion>,
    start: &Adapter,
    backend: &dyn BelnapBackend,
) -> Result<Outcome, RoundError> {
    cfg.validate()
        .map_err(|e| RoundError::Config(e.to_string()))?;
    if contributions.len() < usize::from(cfg.min_participants) {
        return Err(RoundError::TooFew {
            got: contributions.len(),
            need: cfg.min_participants,
        });
    }
    let n_values = delta::value_count(start);
    let chunks = delta::chunk_count(n_values, cfg.chunk_dim);
    let chunks_u32 =
        u32::try_from(chunks).map_err(|_| RoundError::Other("too many chunks".into()))?;
    let rules = cfg.chunk_rules();
    let mut agg: Vec<i64> = Vec::with_capacity(n_values as usize);
    let mut input_hashes = Vec::with_capacity(chunks as usize);
    let mut output_hashes = Vec::with_capacity(chunks as usize);
    let mut in_payloads = Vec::with_capacity(chunks as usize);
    let mut out_payloads = Vec::with_capacity(chunks as usize);
    let mut state_counts = [0u64; 4];
    for c in 0..chunks as usize {
        let rows: Vec<&[i64]> = contributions
            .iter()
            .map(|k| delta::row(&k.artifact.values, cfg.chunk_dim, c))
            .collect();
        let input = chunk_input(&rows, &rules).map_err(|e| RoundError::Chunk {
            chunk: c,
            msg: e.to_string(),
        })?;
        let output = backend.aggregate(&input).await?;
        let dim = rows[0].len();
        let parsed = parse_output(&output, dim).map_err(|e| RoundError::Chunk {
            chunk: c,
            msg: e.to_string(),
        })?;
        for s in &parsed.states {
            state_counts[usize::from(*s)] += 1;
        }
        agg.extend_from_slice(&parsed.values);
        let ih = keccak(&[&input]);
        let oh = keccak(&[&output]);
        input_hashes.push(hex0x(&ih));
        output_hashes.push(hex0x(&oh));
        in_payloads.push(ih);
        out_payloads.push(oh);
    }
    let merged = delta::merge(start, &agg, cfg.value_scale_log2)
        .map_err(|e| RoundError::Other(e.to_string()))?;
    let merged_adapter =
        delta::merged_gguf(start, &merged).map_err(|e| RoundError::Other(e.to_string()))?;
    let adapter_sha256 = sha256(&merged_adapter);

    let part_payloads: Vec<B32> = contributions
        .iter()
        .map(|k| participant_payload(&k.result.worker, &k.result.delta_root))
        .collect();
    let to_err = |e: tree::TreeError| RoundError::Other(e.to_string());
    let participants_root = tree::root(&part_payloads).map_err(to_err)?;
    let input_root = tree::root(&in_payloads).map_err(to_err)?;
    let output_root = tree::root(&out_payloads).map_err(to_err)?;
    let round_id = cfg.round_id(ordinal);
    let config_hash = cfg.config_hash();
    let participants_u16 = u16::try_from(contributions.len())
        .map_err(|_| RoundError::Other("too many participants".into()))?;
    let record = record_digest(
        cfg.chain_id,
        &cfg.ledger,
        &round_id,
        &config_hash,
        &participants_root,
        &input_root,
        &output_root,
        &adapter_sha256,
        n_values,
        chunks_u32,
        participants_u16,
    );
    let participants = contributions
        .into_iter()
        .map(|k| BundleParticipant {
            worker: k.result.worker,
            job: k.job.0.clone(),
            delta_root: k.result.delta_root,
            delta_sha256: k.result.delta_sha256,
            result: k.result,
        })
        .collect();
    Ok(Outcome {
        bundle: Bundle {
            version: BUNDLE_VERSION.into(),
            ordinal,
            round_id,
            config_hash,
            config: cfg.clone(),
            participants,
            n_values,
            chunks: chunks_u32,
            input_hashes,
            output_hashes,
            participants_root,
            input_root,
            output_root,
            adapter_sha256,
            record_digest: record,
            state_counts,
            excluded,
        },
        merged_adapter,
    })
}

/// What a challenger needs for one chunk, read from the bundle and the deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProof {
    pub chunk: u32,
    pub input: String,
    pub input_proof: Vec<String>,
    pub output_hash: String,
    pub output_proof: Vec<String>,
    /// Per participant, for a row challenge.
    pub rows: Vec<RowProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowProof {
    pub participant: u16,
    pub worker: String,
    pub delta_root: String,
    pub participant_proof: Vec<String>,
    pub row_hash: String,
    pub row_proof: Vec<String>,
}

fn parse_list(v: &[String]) -> Result<Vec<B32>, RoundError> {
    v.iter()
        .map(|s| citrate_training_worker::fl::parse_hex::<32>(s).map_err(RoundError::Other))
        .collect()
}

/// Build the proofs for `chunk` of a committed bundle. The input is rebuilt
/// from the delta artifacts and must hash to the bundle's committed leaf, so a
/// proof is only ever produced for the input the round actually committed.
pub fn chunk_proof(b: &Bundle, deltas: &Path, chunk: u32) -> Result<ChunkProof, RoundError> {
    let c = chunk as usize;
    if c >= b.chunks as usize {
        return Err(RoundError::Other(format!("chunk {chunk} >= {}", b.chunks)));
    }
    let mut arts = Vec::with_capacity(b.participants.len());
    for p in &b.participants {
        let path = deltas.join(format!("{}.fld", hex::encode(p.delta_sha256)));
        let bytes = std::fs::read(&path)?;
        if sha256(&bytes) != p.delta_sha256 {
            return Err(RoundError::Other(format!(
                "{} does not hash",
                path.display()
            )));
        }
        arts.push(
            Artifact::decode(&bytes, b.config.max_values)
                .map_err(|e| RoundError::Other(e.to_string()))?,
        );
    }
    let rows: Vec<&[i64]> = arts
        .iter()
        .map(|a| delta::row(&a.values, b.config.chunk_dim, c))
        .collect();
    let input = chunk_input(&rows, &b.config.chunk_rules())
        .map_err(|e| RoundError::Other(e.to_string()))?;
    let ins = parse_list(&b.input_hashes)?;
    let outs = parse_list(&b.output_hashes)?;
    if keccak(&[&input]) != ins[c] {
        return Err(RoundError::Other(
            "the rebuilt input does not match the bundle's committed leaf".into(),
        ));
    }
    let to_hex = |v: Vec<B32>| v.iter().map(|x| hex0x(x)).collect::<Vec<_>>();
    let terr = |e: tree::TreeError| RoundError::Other(e.to_string());
    let part_payloads: Vec<B32> = b
        .participants
        .iter()
        .map(|p| participant_payload(&p.worker, &p.delta_root))
        .collect();
    let mut row_proofs = Vec::with_capacity(arts.len());
    for (i, (p, a)) in b.participants.iter().zip(arts.iter()).enumerate() {
        let rh = delta::row_hashes(&a.values, b.config.chunk_dim)
            .map_err(|e| RoundError::Other(e.to_string()))?;
        row_proofs.push(RowProof {
            participant: i as u16,
            worker: hex0x(&p.worker),
            delta_root: hex0x(&p.delta_root),
            participant_proof: to_hex(tree::proof(&part_payloads, i).map_err(terr)?),
            row_hash: hex0x(&rh[c]),
            row_proof: to_hex(tree::proof(&rh, c).map_err(terr)?),
        });
    }
    Ok(ChunkProof {
        chunk,
        input: hex0x(&input),
        input_proof: to_hex(tree::proof(&ins, c).map_err(terr)?),
        output_hash: hex0x(&outs[c]),
        output_proof: to_hex(tree::proof(&outs, c).map_err(terr)?),
        rows: row_proofs,
    })
}

/// The roots and record digest that follow from a bundle's own leaf lists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Roots {
    #[serde(with = "hexser")]
    pub participants_root: B32,
    #[serde(with = "hexser")]
    pub input_root: B32,
    #[serde(with = "hexser")]
    pub output_root: B32,
    #[serde(with = "hexser")]
    pub record_digest: B32,
}

/// Recompute the roots and record digest from a bundle's leaves, without
/// trusting the roots it states.
pub fn recompute_roots(b: &Bundle) -> Result<Roots, RoundError> {
    let terr = |e: tree::TreeError| RoundError::Other(e.to_string());
    let part_payloads: Vec<B32> = b
        .participants
        .iter()
        .map(|p| participant_payload(&p.worker, &p.delta_root))
        .collect();
    let participants_root = tree::root(&part_payloads).map_err(terr)?;
    let input_root = tree::root(&parse_list(&b.input_hashes)?).map_err(terr)?;
    let output_root = tree::root(&parse_list(&b.output_hashes)?).map_err(terr)?;
    let participants = u16::try_from(b.participants.len())
        .map_err(|_| RoundError::Other("too many participants".into()))?;
    let record_digest = record_digest(
        b.config.chain_id,
        &b.config.ledger,
        &b.round_id,
        &b.config_hash,
        &participants_root,
        &input_root,
        &output_root,
        &b.adapter_sha256,
        b.n_values,
        b.chunks,
        participants,
    );
    Ok(Roots {
        participants_root,
        input_root,
        output_root,
        record_digest,
    })
}

/// A published bundle whose stated roots do not follow from its leaves is
/// malformed.
pub fn check_bundle_roots(b: &Bundle) -> Result<(), RoundError> {
    let r = recompute_roots(b)?;
    for (name, got, want) in [
        ("participants", r.participants_root, b.participants_root),
        ("input", r.input_root, b.input_root),
        ("output", r.output_root, b.output_root),
    ] {
        if got != want {
            return Err(RoundError::Other(format!(
                "{name} root does not follow from its leaves"
            )));
        }
    }
    if r.record_digest != b.record_digest {
        return Err(RoundError::Other(
            "record digest does not follow from the bundle".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    include!("fl_round_tests.rs");
}
