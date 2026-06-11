//! Stage-fault recovery integration test (CM-08 WP-08.2 S1).
//!
//! Scenario: 4-stage pipeline. Stage 2 (w3) faults before
//! receiving activation from stage 1 (w2). MockChain reflects the
//! fault (stage_owner returns None). A replacement worker (w5)
//! is reassigned to stage 2. We then forward the activation to
//! w5 and the pipeline completes.
//!
//! This test drives the resilient-wait path without the in-flight
//! activation being stuck at the upstream sender: the upstream
//! (w2) is responsible for re-forwarding to the new owner after
//! detecting reassignment. For S1 the test focuses on the
//! downstream perspective — stage 3 (w4) uses
//! `serve_request_resilient` and recovers when stage 2 is
//! reassigned mid-wait.

use std::sync::Arc;
use std::time::Duration;

use citrate_training_worker::{
    InProcessTransport, MockPipelineChainClient, PipelineChainClient, PipelineJobSpec,
    PipelineRequestState, PipelineWorker, StageRole, Transport, WorkerMessage,
};
use ethereum_types::{Address, H256};

fn spec() -> PipelineJobSpec {
    PipelineJobSpec {
        stage_count: 4,
        payment_per_request: 4_000_000_000_000_000_000u128,
        per_stake_per_stage: 2_000_000_000_000_000_000u128,
        model_hash: H256::repeat_byte(0x11),
    }
}

#[tokio::test]
async fn stage_fault_then_reassign_resumes_pipeline() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .try_init();

    let chain = MockPipelineChainClient::new();
    let transport = InProcessTransport::new();

    let w1 = Address::repeat_byte(0x01);
    let w2 = Address::repeat_byte(0x02);
    let w3 = Address::repeat_byte(0x03);
    let w4 = Address::repeat_byte(0x04);
    let w5 = Address::repeat_byte(0x05); // replacement for w3
    let requester = Address::repeat_byte(0xAA);

    for w in [w1, w2, w3, w4, w5] {
        transport.register(w).await;
    }

    let job_id = chain.create_active_job(spec(), vec![w1, w2, w3, w4]);
    let req_id = chain.submit_request(job_id, requester).expect("submit");

    // Stage 3 (w4) uses the RESILIENT path with a short timeout +
    // 10 retries so the test can fault+reassign+recover quickly.
    let worker4 = PipelineWorker::new(
        StageRole {
            job_id,
            stage_index: 3,
            self_address: w4,
            total_stages: 4,
        },
        transport.scoped(w4),
        Arc::clone(&chain),
    );
    let h4 = tokio::spawn(async move {
        worker4
            .serve_request_resilient(
                req_id,
                None,
                Duration::from_millis(100),
                20, // generous retry budget
            )
            .await
    });

    // Fault stage 2 (w3) BEFORE stage 3 can receive from it.
    chain.fault_stage(job_id, 2);

    // Wait a bit so worker4's resilient wait has detected the
    // upstream-fault state. (Without this, the test is racy; the
    // 100ms timeout gives multiple recheck cycles.)
    tokio::time::sleep(Duration::from_millis(250)).await;

    // Reassign stage 2 to w5, so a new worker can serve.
    chain.reassign_stage(job_id, 2, w5);

    // Simulate the pipeline progression: w1, w2, w5 (replacement),
    // then w4 completes. We have to drive w1, w2, w5 manually
    // here since the test hadn't spawned them as tokio tasks. For
    // the recovery property, the key observation is that w4 is
    // ABLE to recover when its upstream activation eventually
    // arrives — which happens when we broadcast the expected
    // PipelineActivation targeting stage 3.
    //
    // Advance the chain progress so by the time w5 forwards to
    // w4, the request is at progress 3 (ready for w4 to serve).
    for (stage, worker) in [(0u32, w1), (1, w2), (2, w5)] {
        chain
            .advance_request(req_id, worker)
            .await
            .expect("advance");
        let _ = stage;
    }
    // w5 (as the new stage-2 owner) forwards activation to stage 3 (w4).
    let activation = b"activation-from-w5-to-w4".to_vec();
    transport
        .scoped(w5)
        .broadcast(WorkerMessage::PipelineActivation {
            request_id: req_id,
            from_stage: 2,
            to_stage: 3,
            from_worker: w5,
            payload: activation.clone(),
        })
        .await
        .expect("forward");

    let result = h4.await.expect("w4 join").expect("w4 resilient");
    let out = result.expect("terminal stage returns payload");
    assert_eq!(out.len(), 32, "keccak output is 32 bytes");

    let snap = chain.request_snapshot(req_id).await.expect("snapshot");
    assert_eq!(snap.state, PipelineRequestState::Completed);
    assert_eq!(snap.progress, 4);
}

#[tokio::test]
async fn resilient_path_happy_path_behaves_like_serve_request() {
    // Sanity: when there's no fault, the resilient wrapper should
    // succeed identically to the base path.
    let chain = MockPipelineChainClient::new();
    let transport = InProcessTransport::new();

    let w1 = Address::repeat_byte(0x11);
    let w2 = Address::repeat_byte(0x22);
    let w3 = Address::repeat_byte(0x33);
    let w4 = Address::repeat_byte(0x44);
    let requester = Address::repeat_byte(0xAA);

    for w in [w1, w2, w3, w4] {
        transport.register(w).await;
    }

    let job_id = chain.create_active_job(spec(), vec![w1, w2, w3, w4]);
    let req_id = chain.submit_request(job_id, requester).expect("submit");

    let worker3 = PipelineWorker::new(
        StageRole {
            job_id,
            stage_index: 2,
            self_address: w3,
            total_stages: 4,
        },
        transport.scoped(w3),
        Arc::clone(&chain),
    );
    let h3 = tokio::spawn(async move {
        worker3
            .serve_request_resilient(req_id, None, Duration::from_millis(500), 5)
            .await
    });

    // Wait briefly then drive stages 0, 1, and the activation to 2.
    tokio::time::sleep(Duration::from_millis(20)).await;
    chain.advance_request(req_id, w1).await.unwrap();
    chain.advance_request(req_id, w2).await.unwrap();
    transport
        .scoped(w2)
        .broadcast(WorkerMessage::PipelineActivation {
            request_id: req_id,
            from_stage: 1,
            to_stage: 2,
            from_worker: w2,
            payload: b"from-w2-to-w3".to_vec(),
        })
        .await
        .unwrap();

    let result = h3.await.expect("join").expect("serve");
    // Stage 2 is NOT terminal, so result is None.
    assert!(result.is_none(), "stage 2 is not terminal");
    let snap = chain.request_snapshot(req_id).await.unwrap();
    assert_eq!(snap.progress, 3);
}
