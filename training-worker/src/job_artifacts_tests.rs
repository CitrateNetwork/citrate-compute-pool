// Tests for the honesty core: a job resolves to REAL, VERIFIED artifacts or it
// is refused.
//
// Every test here is about one property — a worker must never train something
// other than what the chain asked for and then get paid as though it had. The
// protocol cannot detect that after the fact (a Merkle root of gradients looks
// the same either way), so these are the checks that make it impossible.

use super::*;
use std::fs;

fn tmpdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "nat-artifacts-{tag}-{}-{}",
        std::process::id(),
        // Distinct per call site so parallel tests never share a directory.
        tag.len()
    ));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("tmpdir");
    p
}

/// Write `bytes` and return the hash the chain would have committed for them.
fn write_addressed(path: &Path, bytes: &[u8]) -> B256 {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, bytes).expect("write");
    keccak_of(bytes)
}

/// Lay down a complete, self-consistent job: weights + manifest at their own
/// content addresses. Returns the store and the two hashes.
fn good_job(tag: &str, sidecar: Option<&str>) -> (ArtifactStore, B256, B256) {
    let root = tmpdir(tag);
    let store = ArtifactStore::new(&root);

    let weights = b"safetensors-bytes-standing-in-for-real-weights";
    let manifest = br#"{"shards":[],"config_hash":"cfg"}"#;

    let model_hash = keccak_of(weights);
    let dataset_hash = keccak_of(manifest);

    write_addressed(&store.model_dir(&model_hash).join("model.safetensors"), weights);
    write_addressed(&store.manifest_path(&dataset_hash), manifest);
    if let Some(s) = sidecar {
        fs::write(store.sidecar_path(&model_hash), s).expect("sidecar");
    }
    (store, model_hash, dataset_hash)
}

// ── The load-bearing refusals ────────────────────────────────────────────

/// THE CENTRAL PROPERTY. Weights that do not hash to the on-chain commitment
/// mean the worker would train a different model and still be paid. Refuse.
#[test]
fn weights_that_do_not_match_the_commitment_are_refused() {
    let (store, model_hash, dataset_hash) = good_job("bad-weights", None);

    // Corrupt the checkpoint in place — the address still says one thing, the
    // bytes now say another. This is exactly a bad fetch or a swapped file.
    fs::write(
        store.model_dir(&model_hash).join("model.safetensors"),
        b"different weights entirely",
    )
    .expect("corrupt");

    let err = store
        .resolve(&model_hash, &dataset_hash)
        .expect_err("mismatched weights must not resolve");
    match err {
        ArtifactError::HashMismatch { what, .. } => assert_eq!(what, "model checkpoint"),
        other => panic!("expected a hash mismatch on the checkpoint, got {other:?}"),
    }
    assert!(
        err.to_string().contains("Refusing to train"),
        "the message must say plainly that training is refused: {err}"
    );
}

/// Same property for the corpus. Training the right model on the wrong data is
/// just as much a lie, and just as invisible on-chain.
#[test]
fn a_dataset_that_does_not_match_the_commitment_is_refused() {
    let (store, model_hash, dataset_hash) = good_job("bad-data", None);
    fs::write(store.manifest_path(&dataset_hash), br#"{"shards":["other"]}"#)
        .expect("corrupt");

    let err = store
        .resolve(&model_hash, &dataset_hash)
        .expect_err("mismatched dataset must not resolve");
    match err {
        ArtifactError::HashMismatch { what, .. } => assert_eq!(what, "dataset manifest"),
        other => panic!("expected a hash mismatch on the manifest, got {other:?}"),
    }
}

/// MIXTURE-OF-EXPERTS IS REFUSED BY NAME.
///
/// NAT rejected learned expert routing in ADR-0001 ("loses interpretability"),
/// and `02_ARCHITECTURE.md` §11 makes "declared zone partitioning … versus
/// learned-from-scratch expert routing" the novelty wedge. There is no MoE
/// trainer in NAT: no gate network, no expert dispatch, no load-balancing loss,
/// no capacity factor.
///
/// The dangerous behaviour would be silently training an MoE job on the dense
/// arm — it would produce a real model, converge, and commit valid roots, while
/// being the wrong architecture. So the refusal is explicit and names the
/// architecture.
#[test]
fn a_mixture_of_experts_job_is_refused_not_silently_trained_as_dense() {
    let (store, model_hash, dataset_hash) =
        good_job("moe", Some(r#"{"architecture":"mixture-of-experts","zones":[]}"#));

    let err = store
        .resolve(&model_hash, &dataset_hash)
        .expect_err("an MoE job must not resolve to the dense arm");
    match &err {
        ArtifactError::UnsupportedArchitecture(a) => assert_eq!(a, "mixture-of-experts"),
        other => panic!("expected UnsupportedArchitecture, got {other:?}"),
    }
    assert!(
        err.to_string().contains("not the one the job asked for"),
        "the refusal must explain WHY substituting is wrong: {err}"
    );
}

/// A sidecar we cannot parse leaves the architecture unknown. Unknown is refused,
/// not defaulted — defaulting to dense would train the H-01 baseline for a zone
/// job and look entirely successful.
#[test]
fn an_unparseable_sidecar_is_refused_rather_than_defaulted() {
    let (store, model_hash, dataset_hash) = good_job("bad-sidecar", Some("{not json"));
    let err = store
        .resolve(&model_hash, &dataset_hash)
        .expect_err("an unreadable sidecar must not resolve");
    assert!(matches!(err, ArtifactError::UnreadableSidecar(_)), "{err:?}");
}

/// Missing artifacts are an honest "not here yet", distinct from a mismatch.
/// Conflating the two would make a corrupted fetch look like a pending one.
#[test]
fn missing_artifacts_report_missing_not_mismatch() {
    let root = tmpdir("missing");
    let store = ArtifactStore::new(&root);
    let err = store
        .resolve(&B256::repeat_byte(0xAA), &B256::repeat_byte(0xBB))
        .expect_err("nothing is on disk");
    assert!(matches!(err, ArtifactError::Missing { .. }), "{err:?}");
}

// ── The architectures that ARE supported ─────────────────────────────────

/// The two real arms in `nat_candle::autoreg`: the zone-partitioned LM and the
/// dense LM. Both resolve; neither is a fallback for the other.
#[test]
fn a_zone_partitioned_job_resolves_to_the_zone_arm() {
    let sidecar = r#"{
        "version": 1,
        "zones": [
            {"id":"SM","core":"Ssm"},  {"id":"CB","core":"Ssm"},
            {"id":"HP","core":"Attention"}, {"id":"PF","core":"Attention"},
            {"id":"CX","core":"Attention"}, {"id":"MX","core":"NonLearned"}
        ]
    }"#;
    let (store, m, d) = good_job("zones", Some(sidecar));
    let a = store.resolve(&m, &d).expect("a valid zone job must resolve");
    assert_eq!(a.architecture, Architecture::ZonePartitioned);
    assert_eq!(a.architecture.as_str(), "zone-partitioned");
}

/// An explicit `"architecture":"dense"` resolves to the dense arm.
#[test]
fn an_explicit_dense_job_resolves_to_the_dense_arm() {
    let (store, m, d) = good_job("dense-explicit", Some(r#"{"architecture":"dense"}"#));
    let a = store.resolve(&m, &d).expect("a valid dense job must resolve");
    assert_eq!(a.architecture, Architecture::Dense);
}

/// A checkpoint with NO sidecar is a plain dense model. This is a supported job
/// shape in its own right, not a guess: `AutoregDenseLm` is a standalone trainer
/// as well as the H-01 equal-parameter baseline.
#[test]
fn a_checkpoint_with_no_sidecar_is_a_dense_job() {
    let (store, m, d) = good_job("dense-implicit", None);
    let a = store.resolve(&m, &d).expect("a bare checkpoint is a dense job");
    assert_eq!(a.architecture, Architecture::Dense);
}

/// Resolved artifacts carry the VERIFIED hashes, so the provenance record can
/// state what was actually trained rather than echoing what was requested.
#[test]
fn resolved_artifacts_carry_the_verified_hashes_and_real_paths() {
    let (store, m, d) = good_job("carries", None);
    let a = store.resolve(&m, &d).expect("resolve");

    assert_eq!(a.model_start_hash, m);
    assert_eq!(a.dataset_hash, d);
    assert!(
        a.checkpoint_dir.join("model.safetensors").exists(),
        "the checkpoint dir must be the one nat_candle::load reads"
    );
    assert!(a.manifest_path.exists(), "the manifest must be a real file");
}

/// Content addressing: the same model named by two jobs is one copy on disk, and
/// the address is derived from the bytes rather than asserted alongside them.
#[test]
fn the_store_is_content_addressed() {
    let root = tmpdir("addressing");
    let store = ArtifactStore::new(&root);
    let a = B256::repeat_byte(0x11);
    let b = B256::repeat_byte(0x22);

    assert_ne!(store.model_dir(&a), store.model_dir(&b));
    assert_eq!(store.model_dir(&a), store.model_dir(&a));
    assert!(
        store.model_dir(&a).to_string_lossy().contains(&hex::encode(a.as_bytes())),
        "the path must contain the hash, so a wrong fetch cannot occupy a right address"
    );
}

// ── The commitment grid (declared per job, read by BOTH sides) ───────────

/// The default is the SAFE grid. `ComputePoolTraining.nextJobId` is still 0, so
/// there are no legacy jobs to preserve — defaulting to the broken grid would
/// buy backwards compatibility with nothing, at the cost of every new job.
#[test]
fn the_default_commitment_grid_is_q16() {
    let (store, m, d) = good_job("grid-default", None);
    let a = store.resolve(&m, &d).expect("resolve");
    assert_eq!(a.commitment_grid, CommitmentGrid::Q16);
}

/// A sidecar with other fields but no `commitment_grid` still defaults to Q16 —
/// the default applies to the FIELD, not to the absence of a sidecar.
#[test]
fn a_sidecar_without_the_field_still_defaults_to_q16() {
    let (store, m, d) = good_job("grid-absent-field", Some(r#"{"architecture":"dense"}"#));
    let a = store.resolve(&m, &d).expect("resolve");
    assert_eq!(a.commitment_grid, CommitmentGrid::Q16);
}

/// The legacy grid is reachable, but only by asking for itByName. Choosing the
/// unsafe grid should be visible in the sidecar where a reviewer sees it, not a
/// silent consequence of omitting a field.
#[test]
fn the_legacy_grid_must_be_asked_for_explicitly() {
    let (store, m, d) = good_job(
        "grid-legacy",
        Some(r#"{"architecture":"dense","commitment_grid":"legacy-f32-scale"}"#),
    );
    let a = store.resolve(&m, &d).expect("resolve");
    assert_eq!(a.commitment_grid, CommitmentGrid::LegacyF32Scale);
    assert_eq!(a.commitment_grid.as_str(), "legacy-f32-scale");
}

/// THE POINT OF THE FIELD. An unrecognised grid is refused, never defaulted.
///
/// Falling back to Q16 for a job that asked for something else would put this
/// worker on a different grid from the challenger — which is precisely the
/// disagreement that gets an honest worker slashed 10%. Better to refuse the job.
#[test]
fn an_unrecognised_commitment_grid_is_refused_not_defaulted() {
    let (store, m, d) = good_job(
        "grid-unknown",
        Some(r#"{"architecture":"dense","commitment_grid":"int8-per-channel"}"#),
    );
    let err = store
        .resolve(&m, &d)
        .expect_err("an unknown grid must not resolve");
    match &err {
        ArtifactError::UnknownCommitmentGrid(g) => assert_eq!(g, "int8-per-channel"),
        other => panic!("expected UnknownCommitmentGrid, got {other:?}"),
    }
    assert!(
        err.to_string().contains("indistinguishable from a dishonest one"),
        "the refusal must explain the consequence of guessing: {err}"
    );
}

/// Grid and architecture are independent axes. A zone job on the legacy grid is
/// a coherent (if unwise) request, and reading one must not silently constrain
/// the other.
#[test]
fn grid_and_architecture_are_independent() {
    let (store, m, d) = good_job(
        "grid-x-arch",
        Some(r#"{"zones":[{"id":"HP"}],"commitment_grid":"legacy-f32-scale"}"#),
    );
    let a = store.resolve(&m, &d).expect("resolve");
    assert_eq!(a.architecture, Architecture::ZonePartitioned);
    assert_eq!(a.commitment_grid, CommitmentGrid::LegacyF32Scale);
}
