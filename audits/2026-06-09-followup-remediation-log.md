---
created: 2026-06-10T00:00:00Z
branch: audit/secrem02-decode-caps
author: Fable 5 (Claude Code)
sprint: SECREM-02-followup-remediation
status: active
repo: citrate-compute-pool
baseline_test_count: 92
---

# citrate-compute-pool — SECREM-02 Remediation Log

> Coverage matrix: `citrate-security/planset/2026-06-10-followup-remediation.md`.

## Phase 4 — WP 4.3 (decode caps)

| Finding | Sev | Red test(s) | Fix (file) | Suite (≥92?) | Disposition |
|---|---|---|---|---|---|
| FUA-COMPUTE-POOL-01 | Med | `forged_signature_is_rejected_by_verify` → oversized-frame branch | `verify_envelope` rejects an over-cap frame up front and decodes both the envelope and inner payload under a `MAX_ENVELOPE_BYTES` (4 MiB) bincode limit (fixint, wire-compatible) — bincode checks length prefixes against the limit, so a hostile mesh message can't pre-allocate GBs before the signature/auth check — `training-worker/src/libp2p_transport.rs` | 93 ✓ | **FIXED** |

## Notes
- Baseline: 92 → 93 (+1 oversized-frame test).
- Used `bincode::DefaultOptions::with_fixint_encoding().with_limit()` to match the
  free-function wire format (verified by existing serialize→verify tests) and avoid
  the deprecated `bincode::config()`.
- Remaining compute-pool: FUA-COMPUTE-POOL-02 + the prior envelope-binding cluster
  (-002…-008) → WP 6.3 (signed EnvelopeContext).
- Branch: `audit/secrem02-decode-caps`.

## Phase 6 — WP 6.3 (signed EnvelopeContext — the binding cluster)

> Branch: `audit/secrem02-envelope-context`. Baseline note: `cargo test
> --workspace` did NOT compile at fcc4f23 — `pool-coordinator` test-only code
> (`spawn_stub_ws`) still passed `String` to tungstenite-0.27 `Message::Text`
> (the 638fd04 adaptation only fixed non-test paths; WP 4.3's "92→93" counted
> training-worker targets only). Fixed first (two `.into()`s, test-only) to
> establish a measurable baseline: **149 passed / 0 failed** workspace-wide
> (93 training-worker + 56 pool-coordinator). Final: **162 passed / 0 failed**.

| Finding | Sev | Red test(s) | Fix (file) | Suite (≥149?) | Disposition |
|---|---|---|---|---|---|
| CITRATE_COMPUTE_POOL-2026-05-31-002 | High | `replayed_envelope_is_rejected_on_second_delivery` (red: replay accepted at fcc4f23) + post-fix tripwires `envelope_context_binding_rejects_cross_scope_replay` (topic/job/epoch/stale/ctx-rewrite), `legacy_v1_envelope_is_rejected` | Wire v2 `WireEnvelope` carries a signed `EnvelopeContext { version, job_id, epoch, nonce, topic, sent_at_ms }`; signature covers domain tag ‖ context ‖ payload; verification fail-closed on version, topic≠receiving topic, job≠scope, epoch outside ±1 of receiver epoch, staleness (5 min) / future skew (30 s), and `(sender, nonce)` replay inside a bounded 4096-entry window (nonce recorded only post-auth). One gossipsub topic PER JOB (`citrate-training/job-<id>`) via new `MeshScope` ctor arg — closes the single-shared-topic deferral. `Transport::set_epoch` (default no-op) lets the worker drive the epoch window — `training-worker/src/libp2p_transport.rs`, `transport.rs`, `worker.rs` | 162 ✓ | **FIXED** |
| CITRATE_COMPUTE_POOL-2026-05-31-003 | Med | `empty_provider_output_fails_job_instead_of_completing` (red: 200+empty output → completeJob at fcc4f23) + `validate_rejects_empty_and_whitespace_output` | `validate_pool_infer_response` rejects empty/whitespace `output` after decode → `ProviderFailed` → `handle_event` routes to `failJob` (buyer refunded) — `pool-coordinator/src/provider.rs` | 162 ✓ | **FIXED** (minimum gate; semantic output commitments remain on-chain INFER-S2 scope, noted in code) |
| CITRATE_COMPUTE_POOL-2026-05-31-004 | Med | `b64_decode_rejects_invalid_char_in_sextet_3_and_4` (red: silent truncation at fcc4f23) | `base64_standard_decode` distinguishes `-1` invalid from `-2` padding in ALL four sextet positions; padding legal only in the final quartet; data-after-padding errors — `training-worker/src/attestation.rs` | 162 ✓ | **FIXED** |
| CITRATE_COMPUTE_POOL-2026-05-31-005 | Low | No direct red (stall reproduction = multi-minute hang); covered by mutation pass on the reused `EpochAggregator` gates (001 tripwires) + integration suite liveness | Coordinator drain: 60 s per-message timeout → loud epoch failure. Non-coordinator archive drain: routed through `EpochAggregator` (same membership + `(worker, step)` dedup + epoch gate as the 001 fix), counts unique commits only, bounded by 60 s per-message + 120 s overall deadline (archive is best-effort → partial archive + proceed) — `training-worker/src/worker.rs`. Transport-level uniqueness additionally enforced by the -002 nonce window | 162 ✓ | **FIXED** |
| CITRATE_COMPUTE_POOL-2026-05-31-006 | Low | `decode_rejects_wrong_event_signature` (ws), `poll_rejects_log_from_wrong_contract` + `poll_rejects_log_with_foreign_topic0` (http) — all red at fcc4f23 — plus `decode_rejects_wrong_emitting_address`, `decode_rejects_missing_dedup_fields` | Both decoders re-verify `topics[0]` == ComputeRequested sig AND `log.address` == configured pool contract before decoding; dedup-key fields (`transactionHash`, `logIndex`, `blockNumber`) are now hard decode errors instead of `unwrap_or` defaults (blockNumber also feeds the -008 election) — `pool-coordinator/src/ws_chain.rs`, `http_chain.rs` | 162 ✓ | **FIXED** |
| CITRATE_COMPUTE_POOL-2026-05-31-007 | Low | No behavioral red (timing oracle not observable in unit tests); keystore decrypt regression tests stay green | MAC comparison via `subtle::ConstantTimeEq::ct_eq` in BOTH byte-identical wallet copies (`pool-coordinator/src/wallet.rs`, `training-worker/src/wallet.rs`); `subtle = "2.5"` added (already in the dep tree via k256/aes) | 162 ✓ | **FIXED** (HYG-DUP wallet unification remains open, tracked in audit hygiene notes) |
| CITRATE_COMPUTE_POOL-2026-05-31-008 | Low | `coordinator_check_uses_event_block_epoch` (red: epoch 0 queried for block 250 at fcc4f23) | `handle_event` derives `epoch = epoch_of(event.block_number)` (mirrors `block.number / EPOCH_LENGTH` in ComputePool.sol); `block_number` is now a required log field (-006) — `pool-coordinator/src/lib.rs` | 162 ✓ | **FIXED** |
| FUA-COMPUTE-POOL-02 | Med | `forged_activation_from_non_owner_is_ignored` (red: attacker races honest upstream and wins at fcc4f23) + `sender_binding_covers_step_commits_and_activations` | `PipelineActivation` gains `from_worker`; libp2p boundary drops activations whose `from_worker` ≠ verified envelope signer (`enforce_sender_binding`, same rule as `StepCommitted.worker`); `serve_request` additionally requires `from_stage == own stage − 1` AND `from_worker == on-chain stage_owner(from_stage)`, skipping (not erroring on) forged activations so an attacker cannot grief the request — `training-worker/src/transport.rs`, `libp2p_transport.rs`, `pipeline.rs` | 162 ✓ | **FIXED** |

### Mutation pass (drop-the-check → named test must fail; all restored)

| # | Mutation | Killed by |
|---|---|---|
| 1 | drop topic binding (`ctx.topic != expected_topic`) | `envelope_context_binding_rejects_cross_scope_replay` FAILED |
| 2 | drop job binding (`ctx.job_id != expected_job`) | same test FAILED |
| 3 | drop epoch window check | same test FAILED |
| 4 | drop nonce replay check | `replayed_envelope_is_rejected_on_second_delivery` FAILED |
| 5 | verify signature over bare payload (drop context from signed bytes) | 4 tests FAILED incl. `forged_signature_is_rejected_by_verify` |
| 6 | drop `PipelineActivation` arm in `enforce_sender_binding` | `sender_binding_covers_step_commits_and_activations` FAILED |
| 7 | drop stage-owner check in `serve_request` | `forged_activation_from_non_owner_is_ignored` FAILED |
| 8 | revert epoch to hardcoded 0 | `coordinator_check_uses_event_block_epoch` FAILED |
| 9 | drop empty-output validation | `validate_rejects_empty_and_whitespace_output` + `empty_provider_output_fails_job_instead_of_completing` FAILED |
| 10 | drop topic0 check (http decoder) | `poll_rejects_log_with_foreign_topic0` FAILED |
| 11 | drop b64 invalid-char rejection | `b64_decode_rejects_invalid_char_in_sextet_3_and_4` FAILED |

## Notes (WP 6.3)
- **Wire-format versioning order**: fixtures were re-signed to v2 (helper
  `signed_envelope` + `test_ctx`) in the same change that enforces v2 —
  mirroring agent-runtime 3.4's "version + re-sign fixtures before enforce".
  v1 envelopes are rejected outright (`legacy_v1_envelope_is_rejected`):
  there is no deployed v1 mesh to interoperate with, so fail-closed beats a
  compat window.
- **WP 4.3 decode caps untouched**: `MAX_ENVELOPE_BYTES` / `bincode_limited`
  remain the first gate in `verify_envelope`; the oversized-frame branch of
  `forged_signature_is_rejected_by_verify` still passes.
- Replay-window limits: nonces are only recorded after signature verification,
  so unauthenticated traffic cannot evict window entries; long-range replay
  beyond the 4096-entry window is cut by the 5-minute freshness bound.
- In-process transports (`InProcessTransport`/`ScopedTransport`) are trusted
  test seams: the FUA-02 `from_worker` spoof is enforced at the libp2p
  boundary; the chain-side `stage_owner` check additionally holds on ALL
  transports.
- clippy: CI gates `-D warnings`; local toolchain shows 9 pre-existing lib
  warnings at fcc4f23 vs 8 after this WP (net −1; none introduced).
- Pre-existing workspace test-compile break fixed first (commit on this
  branch, test-only `.into()`s in `ws_chain.rs` stub) — without it no
  workspace baseline was measurable.
