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
