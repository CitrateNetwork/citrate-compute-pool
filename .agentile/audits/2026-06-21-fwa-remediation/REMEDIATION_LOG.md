---
created: 2026-06-21T00:00:00Z
branch: remediation/fwa-2026-06
author: "Claude Opus 4.8 (1M context) — RM-MISC/compute-pool remediation agent"
standard: Agentile-Audit Standard v0.2
audit_id: 2026-06-20-federation-wide-audit
chunk: FWA-C8
repo: citrate-compute-pool
pinned_pre_fix_sha: ae9358db140a0c210fcb1ff700dd8b8627d5e612
---

# FWA-C8 remediation log — `citrate-compute-pool`

Closes **FWA-C8-01** (MEDIUM) from the federation-wide audit. Red-test-driven,
fix-with-proof. FWA-C8-02 (INFO, redirect cap) is in `citrate-node-agent`, a
different repo — out of scope for this compute-pool pass.

## FWA-C8-01 — Coordinator outbound endpoints accept plaintext `http://` with no TLS/scheme validation

**MEDIUM · class F (isolation) · asset=compute, impact=integrity**

### Finding recap
The pool-coordinator made outbound requests over operator-configured URLs with
no TLS/scheme validation, unlike `citrate-node-agent` which already enforces
`chainio::outbound::validate_outbound_url` (https any-host / http loopback-only)
on all its outbound clients. A plaintext-remote member endpoint or RPC was
silently accepted; a network MITM on a plaintext leg can read the buyer prompt
and forge a completion (coordinator pays for fabricated work via `completeJob`),
forge a failure (griefs an honest member via `failJob`), or — on the RPC/ws leg
— feed false chain-truth (`coordinatorFor` / `pool_members` / receipts /
`ComputeRequested` events).

### RED (reproduce-or-retract)
Promoted `evidence/FWA-C8-01-red.rs` to an in-tree `#[test]` in its
fixed-tripwire shape:
`pool-coordinator/src/config.rs::tests::red_fwa_c8_01_remote_plaintext_member_endpoint_is_refused`.

- **Pre-fix:** FAILED — `parse_endpoints("0xaa..=http://attacker.example/infer")`
  returned `Ok`, storing the remote-plaintext endpoint verbatim (would be the
  dispatch target). Panic at `config.rs:132` (`is_err()` assertion).
- **Post-fix:** PASSES — the endpoint is refused fail-closed.

### FIX
Ported node-agent's outbound gate into the coordinator (separate workspace, no
shared `chainio` dep, so mirrored one-for-one) and applied it at **every**
outbound-construction site:

| Site | Where | Gate |
|---|---|---|
| member dispatch endpoints | `config.rs::parse_endpoints` (per value, parse time) | `validate_outbound_url` |
| JSON-RPC URL | `config.rs::CoordinatorConfig::from_env` (config load) | `validate_outbound_url` |
| JSON-RPC adapter boundary | `http_chain.rs::HttpChainAdapter::try_new` (defense-in-depth) | `validate_outbound_url` |
| ws event subscription (SWEEP-found) | `ws_chain.rs::WsChainSubscriber::connect` (before dial) | `validate_outbound_ws_url` |

New module: `pool-coordinator/src/outbound.rs` — `validate_outbound_url[_with]`
(http/https) + `validate_outbound_ws_url[_with]` (ws/wss), `OutboundUrlError`
{UnsupportedScheme, PlaintextNonLoopback, Malformed}, host extraction (rejects
userinfo, empty host, unbracketed-IPv6, non-numeric port), loopback detection.
Dev-only escape hatch `CITRATE_POOL_ALLOW_INSECURE_OUTBOUND=1` (logged loudly).

`main.rs` switched to `HttpChainAdapter::try_new` (new exit code 7 on a rejected
RPC). Stale `config.rs` doc-comment claiming "HTTPS endpoint" (the gap the audit
called out) corrected to state the gate is now enforced.

Rule applied: `https`/`wss` → any host; `http`/`ws` → loopback only
(`localhost`, `127.0.0.0/8`, `[::1]`); anything else → refused fail-closed at
config-load / connect time so a misconfigured daemon dies at startup, not
mid-job. Default RPC (`http://127.0.0.1:18545`) is loopback → unaffected.

### SWEEP (variant coverage)
Swept every outbound URL/connection in the coordinator + pool crate:

| Sink | Status |
|---|---|
| `config.rs` member_endpoints | GATED (parse time) |
| `config.rs` rpc_url | GATED (config load) |
| `http_chain.rs` `HttpChainAdapter` reqwest client (`.post(&rpc_url)`) | GATED via `try_new` (main.rs); the bare `new` retained for loopback test fixtures only |
| `lib.rs` `dispatch_to_member` reqwest POST | endpoint comes only from the now-gated `member_endpoints` map — no other source |
| `provider.rs` `.post(endpoint)` | same — endpoint is the gated map value |
| **`ws_chain.rs` `connect_async(rpc_ws_url)`** | **NOT in the original finding** — surfaced by the sweep. The opt-in `CITRATE_POOL_WS_URL` event-subscription leg feeds `ComputeRequested` into the dispatch loop = same MITM-able chain-truth vector. Now GATED with `validate_outbound_ws_url`. |
| `metrics.rs` | inbound `/metrics` listener only — not outbound |
| test-harness `accept_async` / stub servers | server-side, not a client dial — out of scope |

### TRIPWIRE (permanent)
1. **In-tree tests** (count monotone non-decreasing): the RED test (now green) +
   `parse_endpoints_accepts_https_and_loopback_http` +
   `http_chain::tests::try_new_refuses_plaintext_remote_rpc` + the full
   `outbound::tests` suite (https/loopback/refuse/malformed/override/ws/host/Display/env).
2. **Semgrep**: `.agentile/audits/2026-06-21-fwa-remediation/semgrep/fwa-c8-01-outbound-tls-gate.yaml`
   — two rules: any `connect_async` or member-endpoint `insert` in a function
   that does not call the gate is an ERROR. Validated:
   - violation fixture → flags exactly the 2 unguarded sinks;
   - guarded fixture → 0 findings;
   - **production `pool-coordinator/src` → 0 findings** (the fixed code passes).

### MUTATION
`cargo mutants -p citrate-pool-coordinator -f pool-coordinator/src/outbound.rs`:
- First pass: 17/24 caught (71%) — 7 missed (env-wrapper `==`, ws-wrapper body,
  Display fmt, `host_of` empty-host / userinfo / multi-colon edges).
- Added 5 targeted kill-tests (empty-host-with-port, userinfo, unbracketed
  multi-colon, Display specificity, env-override-exactly with a process-wide
  `ENV_TEST_LOCK` so the override can't leak across parallel tests).
- **Second pass: 24/24 caught = 100% (≥90% gate met).**

### Test-count delta
- Pre-fix: 63 (55 lib + 8 smoke).
- Post-fix: 78 (70 lib + 8 smoke). **+15**, monotone non-decreasing, no existing
  test weakened. Two pre-existing config fixtures had their endpoint URLs
  changed from `http://m1/infer` (remote plaintext) to `https://...` /
  loopback — adapted to the new contract, not weakened (they still assert the
  same parse cardinality).

### Honesty
GREEN. Red→green confirmed by direct test execution. Mutation 100% by direct
`cargo mutants` run. Semgrep validated against both fixtures + production source
with `semgrep 1.136.0` (installed this session under
`~/Library/Python/3.9/bin`). No NEEDS-REPRO remaining for FWA-C8-01.

### Files + LOC
- `pool-coordinator/src/outbound.rs` — NEW, ~330 LOC (module + tests)
- `pool-coordinator/src/config.rs` — gate at parse + load; doc fix; +3 tests
- `pool-coordinator/src/http_chain.rs` — `try_new` + tripwire test
- `pool-coordinator/src/ws_chain.rs` — gate in `connect`
- `pool-coordinator/src/main.rs` — `try_new` wiring + exit code 7
- `pool-coordinator/src/lib.rs` — `pub mod outbound;`
- `.agentile/audits/2026-06-21-fwa-remediation/semgrep/` — rule + 2 fixtures
