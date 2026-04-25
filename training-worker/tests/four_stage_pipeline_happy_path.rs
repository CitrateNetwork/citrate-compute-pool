//! CM-08 WP-08.2 S0 integration test — 4 stages serve one request.
//!
//! Spins up 4 pipeline workers as tokio tasks sharing an
//! `InProcessTransport` + `MockPipelineChainClient`. Stage 0
//! receives the request prompt, forwards activation to stage 1,
//! ..., stage 3 emits the final output. Each stage calls
//! `advance_request` on the mock chain after serving.
//!
//! Asserts:
//!   - All four stages return successfully
//!   - Final output is non-empty (keccak-chain through 4 stages)
//!   - On-chain request state reaches Completed
//!   - Each worker earned payment/4 on-chain

use std::sync::Arc;

use citrate_training_worker::{
    pipeline::PipelineRequestState, InProcessTransport, MockPipelineChainClient,
    PipelineChainClient, PipelineJobSpec, PipelineWorker, StageRole, Transport,
};
use ethereum_types::{Address, H256};

fn spec() -> PipelineJobSpec {
    PipelineJobSpec {
        stage_count: 4,
        payment_per_request: 4_000_000_000_000_000_000u128, // 4 ether
        per_stake_per_stage: 2_000_000_000_000_000_000u128, // 2 ether
        model_hash: H256::repeat_byte(0x11),
    }
}

#[tokio::test]
async fn four_stage_pipeline_happy_path() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .try_init();

    let chain = MockPipelineChainClient::new();
    let transport = InProcessTransport::new();

    let w0 = Address::repeat_byte(0x01);
    let w1 = Address::repeat_byte(0x02);
    let w2 = Address::repeat_byte(0x03);
    let w3 = Address::repeat_byte(0x04);
    let requester = Address::repeat_byte(0xAA);

    // Register each worker on the transport so broadcasts reach them.
    for w in [w0, w1, w2, w3] {
        transport.register(w).await;
    }

    // Create the job + submit one request.
    let job_id = chain.create_active_job(spec(), vec![w0, w1, w2, w3]);
    let req_id = chain.submit_request(job_id, requester).expect("submit");

    // Construct each stage's worker with a sender-scoped transport
    // handle (so broadcasts skip the sender's own queue).
    let worker0 = PipelineWorker::new(
        StageRole {
            job_id,
            stage_index: 0,
            self_address: w0,
            total_stages: 4,
        },
        transport.scoped(w0),
        Arc::clone(&chain),
    );
    let worker1 = PipelineWorker::new(
        StageRole {
            job_id,
            stage_index: 1,
            self_address: w1,
            total_stages: 4,
        },
        transport.scoped(w1),
        Arc::clone(&chain),
    );
    let worker2 = PipelineWorker::new(
        StageRole {
            job_id,
            stage_index: 2,
            self_address: w2,
            total_stages: 4,
        },
        transport.scoped(w2),
        Arc::clone(&chain),
    );
    let worker3 = PipelineWorker::new(
        StageRole {
            job_id,
            stage_index: 3,
            self_address: w3,
            total_stages: 4,
        },
        transport.scoped(w3),
        Arc::clone(&chain),
    );

    // Input prompt for stage 0.
    let prompt = b"Explain pipeline parallelism in two sentences.".to_vec();

    // Spawn all four workers concurrently.
    let h3 = tokio::spawn(async move { worker3.serve_request(req_id, None).await });
    let h2 = tokio::spawn(async move { worker2.serve_request(req_id, None).await });
    let h1 = tokio::spawn(async move { worker1.serve_request(req_id, None).await });
    let h0 = tokio::spawn(async move { worker0.serve_request(req_id, Some(prompt)).await });

    let (r0, r1, r2, r3) = tokio::join!(h0, h1, h2, h3);

    let out0 = r0.expect("w0 join").expect("w0 serve");
    let out1 = r1.expect("w1 join").expect("w1 serve");
    let out2 = r2.expect("w2 join").expect("w2 serve");
    let out3 = r3.expect("w3 join").expect("w3 serve");

    // Only the terminal stage returns the final activation.
    assert!(out0.is_none(), "stage 0 has a next stage");
    assert!(out1.is_none(), "stage 1 has a next stage");
    assert!(out2.is_none(), "stage 2 has a next stage");
    let final_output = out3.expect("stage 3 terminal must return output");
    assert_eq!(final_output.len(), 32, "keccak output is 32 bytes");

    // On-chain state: Completed with every worker paid.
    let snap = chain.request_snapshot(req_id).await.expect("snapshot");
    assert_eq!(snap.state, PipelineRequestState::Completed);
    assert_eq!(snap.progress, 4);

    let per_stage = 1_000_000_000_000_000_000u128; // 4 ether / 4 stages
    for w in [w0, w1, w2, w3] {
        assert_eq!(
            chain.payment_earned(req_id, w),
            per_stage,
            "each worker earns 1 ether"
        );
    }
}
