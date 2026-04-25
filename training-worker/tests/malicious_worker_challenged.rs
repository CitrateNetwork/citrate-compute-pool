//! Challenge-flow integration test (CM-07 WP-07.4 S0).
//!
//! Scenario:
//!   - 3 workers, w1/w2/w3. w1 is coordinator.
//!   - w2 runs `MaliciousTinyModel` targeting (epoch 0, step 0).
//!   - All three workers compute their step commits. w2's is
//!     tampered; w1/w3's are honest.
//!   - Coordinator (w1) aggregates all commits into a Merkle root,
//!     posts it on-chain via `commitEpoch`.
//!   - A challenger (external verifier, address `challenger`)
//!     recomputes w2's step commit honestly, compares with w2's
//!     published commit, detects the mismatch.
//!   - Challenger builds a Merkle proof for w2's tampered leaf
//!     against the stored epoch root, calls `challenge_step` with
//!     bond.
//!   - Committee (3 members) votes Uphold × 2 → resolution fires.
//!   - w2 is slashed SLASH_BPS (10%) of stake; challenger receives
//!     bond + half-slash.
//!
//! Asserts:
//!   - `challenge_step` accepts the proof (proving off-chain merkle
//!     construction matches the on-chain verifier byte-for-byte)
//!   - After quorum, w2's `worker_total_slashed` = 10% of stake
//!   - Challenger reward = bond + 5% of w2's stake

use std::sync::Arc;

use citrate_training_worker::{
    merkle::{compute_leaf, compute_proof},
    ChainClient, DeterministicTinyModel, MaliciousTinyModel, MockChainClient, ModelBackend,
    StepCommit, TrainingJobSpec,
};
use ethereum_types::{Address, H256};

fn spec() -> TrainingJobSpec {
    TrainingJobSpec {
        model_start_hash: H256::repeat_byte(0x11),
        dataset_hash: H256::repeat_byte(0x22),
        epoch_count: 1,
        steps_per_epoch: 2,
        min_workers: 3,
        max_workers: 5,
        challenge_window_blocks: 50,
        per_epoch_budget: 30_000_000_000_000_000_000u128,
        per_worker_stake: 10_000_000_000_000_000_000u128,
    }
}

const CHALLENGE_BOND: u128 = 1_000_000_000_000_000_000u128; // 1 ether

/// Compute a worker's full epoch 0 step commits using a given backend.
/// Returns the list of StepCommit values in order.
async fn compute_commits_for_worker<M: ModelBackend>(
    backend: &M,
    starting: H256,
    worker: Address,
    shard_index: u32,
    steps_per_epoch: u32,
) -> Vec<StepCommit> {
    let mut commits = Vec::new();
    let mut prev_weights = starting;
    for step in 0..steps_per_epoch {
        let result = backend
            .forward_backward(prev_weights, 0, step, shard_index)
            .await
            .expect("forward");
        let commitment = backend.compute_step_commitment(&result.gradients);
        commits.push(StepCommit {
            epoch: 0,
            step,
            worker,
            commitment,
            prev_weights,
        });
        prev_weights = result.post_weights_hash;
    }
    commits
}

#[tokio::test]
async fn malicious_worker_is_challenged_and_slashed() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let sp = spec();
    let job_id = chain.create_job(sp.clone());

    let w1 = Address::repeat_byte(1); // coordinator + honest
    let w2 = Address::repeat_byte(2); // MALICIOUS at (epoch 0, step 0)
    let w3 = Address::repeat_byte(3); // honest
    let challenger = Address::repeat_byte(0xCC);

    // Committee members for vote resolution.
    let c1 = Address::repeat_byte(0xC1);
    let c2 = Address::repeat_byte(0xC2);
    chain.set_committee_member(c1, true).await;
    chain.set_committee_member(c2, true).await;

    for w in [w1, w2, w3] {
        chain
            .join_training_job(job_id, w, sp.per_worker_stake)
            .await
            .expect("join");
    }
    chain.close_recruitment(job_id, w1).await.expect("close");

    // Each worker runs its model, producing step commits.
    let honest = DeterministicTinyModel::new();
    let malicious = MaliciousTinyModel::new(0, 0);

    let w1_commits =
        compute_commits_for_worker(&honest, sp.model_start_hash, w1, 0, sp.steps_per_epoch).await;
    let w2_commits = compute_commits_for_worker(
        &malicious,
        sp.model_start_hash,
        w2,
        1,
        sp.steps_per_epoch,
    )
    .await;
    let w3_commits =
        compute_commits_for_worker(&honest, sp.model_start_hash, w3, 2, sp.steps_per_epoch).await;

    // The honest computation for what w2's step-0 should have been
    // (used by the challenger to verify the miscommit, then by the
    // committee to vote Uphold). For the test we don't need to
    // actually pass this to the contract — the committee's vote is
    // the social proof. But we assert the honest and malicious
    // commits DIFFER, so the challenge is meaningful.
    let w2_honest_step0 =
        compute_commits_for_worker(&honest, sp.model_start_hash, w2, 1, sp.steps_per_epoch)
            .await
            .into_iter()
            .next()
            .expect("w2 honest step 0");
    let w2_actual_step0 = w2_commits.first().expect("w2 actual step 0").clone();
    assert_ne!(
        w2_honest_step0.commitment, w2_actual_step0.commitment,
        "malicious backend must produce different commitment"
    );

    // Coordinator aggregates all commits into a Merkle root.
    let mut all_commits = Vec::new();
    all_commits.extend_from_slice(&w1_commits);
    all_commits.extend_from_slice(&w2_commits);
    all_commits.extend_from_slice(&w3_commits);
    let (epoch_root, leaves) =
        citrate_training_worker::compute_epoch_root(&all_commits);

    // Post the root. Because the coordinator is the one whose local
    // view of all commits includes w2's tampered one, the epoch root
    // hard-commits to it.
    chain
        .commit_epoch(job_id, w1, 0, epoch_root)
        .await
        .expect("commit epoch 0");

    // === Challenger's job ===
    // 1. Recompute w2's honest step 0 (already done above).
    // 2. Pull w2's actual step 0 commit from the mesh/archive
    //    (modeled here as direct access to w2_actual_step0).
    // 3. Compute the leaf for w2's actual commit.
    let disputed_leaf = compute_leaf(&w2_actual_step0);

    // 4. Build the sorted leaves (the convention the coordinator
    //    used). The leaf we're disputing must be present.
    let mut sorted: Vec<&StepCommit> = all_commits.iter().collect();
    sorted.sort_by(|a, b| a.step.cmp(&b.step).then_with(|| a.worker.cmp(&b.worker)));
    let target_index = sorted
        .iter()
        .position(|c| {
            c.epoch == w2_actual_step0.epoch
                && c.step == w2_actual_step0.step
                && c.worker == w2_actual_step0.worker
        })
        .expect("w2's step 0 must be in the tree");

    // 5. Produce the Merkle proof.
    let proof = compute_proof(&leaves, target_index);

    // 6. Open the challenge.
    chain
        .challenge_step(
            job_id,
            challenger,
            0,
            0,
            w2,
            disputed_leaf,
            proof.clone(),
            CHALLENGE_BOND,
        )
        .await
        .expect("challenge must be accepted");

    // === Committee votes ===
    chain
        .vote_challenge(job_id, c1, 0, 0, w2, true)
        .await
        .expect("c1 vote");
    chain
        .vote_challenge(job_id, c2, 0, 0, w2, true)
        .await
        .expect("c2 vote — quorum triggers resolution");

    // === Assertions ===
    let expected_slash = sp.per_worker_stake * 1000 / 10_000; // 10% = 1 ether
    let w2_slash = chain.worker_total_slashed(job_id, w2).await;
    assert_eq!(w2_slash, expected_slash, "target slash applied");

    let expected_reward = CHALLENGE_BOND + expected_slash / 2; // bond + half-slash
    let chal_reward = chain.challenger_reward(challenger).await;
    assert_eq!(chal_reward, expected_reward, "challenger reward correct");
}

#[tokio::test]
async fn bad_proof_is_rejected_without_touching_state() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let sp = spec();
    let job_id = chain.create_job(sp.clone());

    let w1 = Address::repeat_byte(1);
    let w2 = Address::repeat_byte(2);
    let w3 = Address::repeat_byte(3);
    let challenger = Address::repeat_byte(0xCC);

    for w in [w1, w2, w3] {
        chain
            .join_training_job(job_id, w, sp.per_worker_stake)
            .await
            .unwrap();
    }
    chain.close_recruitment(job_id, w1).await.unwrap();

    // Commit a real root but submit a proof for an unrelated leaf.
    let honest = DeterministicTinyModel::new();
    let w1_commits =
        compute_commits_for_worker(&honest, sp.model_start_hash, w1, 0, sp.steps_per_epoch).await;
    let (root, _) = citrate_training_worker::compute_epoch_root(&w1_commits);
    chain.commit_epoch(job_id, w1, 0, root).await.unwrap();

    let bogus_leaf = H256::repeat_byte(0xAB);
    let empty_proof = Vec::<H256>::new();

    let err = chain
        .challenge_step(job_id, challenger, 0, 0, w2, bogus_leaf, empty_proof, CHALLENGE_BOND)
        .await
        .expect_err("bad proof must reject");
    assert!(matches!(
        err,
        citrate_training_worker::ChainError::BadMerkleProof
    ));
}

#[tokio::test]
async fn self_challenge_rejected() {
    let chain: Arc<MockChainClient> = MockChainClient::new();
    let sp = spec();
    let job_id = chain.create_job(sp.clone());

    let w1 = Address::repeat_byte(1);
    let w2 = Address::repeat_byte(2);
    let w3 = Address::repeat_byte(3);
    for w in [w1, w2, w3] {
        chain
            .join_training_job(job_id, w, sp.per_worker_stake)
            .await
            .unwrap();
    }
    chain.close_recruitment(job_id, w1).await.unwrap();

    let honest = DeterministicTinyModel::new();
    let commits =
        compute_commits_for_worker(&honest, sp.model_start_hash, w1, 0, sp.steps_per_epoch).await;
    let (root, leaves) = citrate_training_worker::compute_epoch_root(&commits);
    chain.commit_epoch(job_id, w1, 0, root).await.unwrap();

    // w2 attempts to challenge themselves.
    let proof = compute_proof(&leaves, 0);
    let err = chain
        .challenge_step(job_id, w2, 0, 0, w2, leaves[0], proof, CHALLENGE_BOND)
        .await
        .expect_err("self challenge must reject");
    assert!(matches!(
        err,
        citrate_training_worker::ChainError::SelfChallenge
    ));
}
