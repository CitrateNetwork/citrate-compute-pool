# CP-B-004 — HELD (on-chain / consensus scope)

**Finding:** the pool coordinator submits `completeJob` (which releases the
buyer's escrowed payment to the pool) on the sole evidence of an HTTP 200 whose
`output` field contains one non-whitespace character. There is no proof of
compute of any kind in this repo.

**RED reproduced** (`cp_b_004_RED.txt`): the literal string `"x"` returned to a
4096-token prompt passes `validate_pool_infer_response` — the only gate before
`chain.complete_job()` at `pool-coordinator/src/lib.rs:170-184`.

## Why this is HELD, not fixed off-chain

A sound fix requires proof that the compute was actually performed, and that
proof cannot be established in this repo:

1. **The attacker already holds the member key.** The rated actor is an
   on-chain-`active` pool member present in the operator's `member_endpoints`
   map. Requiring the member to *sign* `(job_id, keccak(output))` — the natural
   off-chain hardening — does not stop them: they own the key, so they sign `"x"`
   and are still paid. Signature binding adds accountability but does not remove
   "member returns garbage → paid", which is the finding.

2. **Proof-of-compute is on-chain scope.** The real remediation is an output
   commitment published with `completeJob` plus a challenge/re-execution/slash
   path — exactly what `provider.rs:72-75` names as "on-chain INFER-S2 scope".
   `ComputePool.sol` is **not in this repo** (MAP §8 coverage limit): `completeJob`
   takes only `job_id` (`http_chain.rs:690-696`), the selected member's address is
   never sent on-chain (`record_dispatch` discards it), so the chain cannot even
   attribute — let alone verify — the work. Adding a commitment/challenge means
   changing the contract and the member `/pool-infer` protocol together.

3. **Any off-chain-only gate is a fig-leaf against the rated actor.** Tightening
   `validate_pool_infer_response` (length, token-count, JSON-shape checks) raises
   the bar for a lazy defector but a member who runs a two-line server can satisfy
   any structural check without doing the work. Inventing such a gate would
   misrepresent the fix, contrary to the task's "don't invent a change".

## Disposition

**HELD** — remediation is consensus/on-chain (ComputePool.sol output commitment +
challenge/slash + member-signed response protocol), to land with the contract
work, not as an off-chain change to this repo. The existing empty-output gate
(`validate_pool_infer_response`, SECREM-02 6.3) remains as the one vector it does
close (pay-for-empty-work); it is left untouched.

Tracked for the on-chain batch. This matches the finding's own severity note:
graded HIGH (not CRITICAL) precisely because settlement soundness depends on
`ComputePool.sol`, which is outside this repository.
