---
created: 2026-07-30
branch: feat/nat-training-backend
author: Claude (Opus 5), directed by @SaulBuilds
status: plan — the commitment grid is settled and cheaper than feared; the two
  gaps both have concrete owners and one is a port, not a build
relates:
  - TRACK_A_GRADING_2026-07-29.md
  - nat/docs/SETTLEMENT_SEAM.md
  - nat/PLANSET/02_ARCHITECTURE.md §11 (the novelty wedge)
---

# Can the co-op live on this backend? — the plan

Short answer: **yes, and the expensive-looking part is not expensive.** The
commitment grid needs no contract change. Of the two gaps, one is a port of code
that already exists and the other turns out not to be a gap for the co-op at all.

---

## 1. The commitment grid — SETTLED, no contract change

The worry was that fixing the grid meant touching a deployed, money-handling
contract. It does not, and the reason is worth writing down because it is not
obvious from the outside.

**`ComputePoolTraining` never recomputes a commitment.** `challengeStep` does
exactly one cryptographic check —
`_verifyMerkleProof(merkleProof, epochRoot, leaf)` — Merkle *inclusion*. The only
`keccak256` calls in the whole contract are the Merkle path's own leaf/internal
prefixes. Resolution is then `voteChallenge`, gated on `committee[msg.sender]`:
**a committee vote, not on-chain recomputation.**

So the quantization grid is **off-chain policy**. The leaf format
(`keccak256(abi.encode(epoch, step, target, stepCommitHash, prevWeightsHash))`)
is unchanged; only what goes *into* `stepCommitHash` moves.

### What that means practically

| | |
|---|---|
| Contract change needed | **None** |
| What must agree | The worker and the committee, **per job** |
| Failure if they disagree | Honest workers challenged and slashed 10% |

The grid therefore has to be a **declared, per-job property** — read from the
same place by the worker that commits and the committee that recomputes. The
natural home is the sidecar, which already declares the architecture and which
`job_artifacts.rs` already parses.

### Done in this PR

- `to_q16` calls `citrate_fed_types::Q16::from_f32` — the *same kernel*
  `nat-types` re-exports and the chain's `0x0110` path uses. "Same grid" is now
  structural, not three reimplementations that happen to agree.
- Pinning that dep forced a real fix: this repo floated on `channel = "stable"`
  (1.93.0 locally) while `citrate-chain`, `nat` and `citrate-fed-types` all pin
  **1.96.0**. The shared grid was literally unreachable from here. Now pinned to
  match — which also unblocks the `nat` dep the backend needs next.
- Both failure forms of the old grid are pinned as tests, found by probing the
  real functions: the f32 scale moves the commitment even with a byte-identical
  payload, and — worse — the shared scale *couples coordinates*, so a one-ULP
  change to element 0 flips element 1.

### Remaining (small)

1. Add `commitment_grid: "q16" | "legacy-f32-scale"` to the sidecar, defaulting
   to `q16` for NAT jobs and `legacy-f32-scale` for anything already running.
2. `NatBackend` overrides `compute_step_commitment` to the Q16 path.
3. The challenger/committee tooling reads the same field. **This is the one that
   matters** — a worker on Q16 and a committee on f32 is worse than both being
   on f32.

---

## 2. Gap: `ToyKeyedSigner` — BLOCKED UPSTREAM, and the reason matters

**Revised after attempting it.** I said this was a port of the existing operator
signer. The signer does exist and is the right one — but the two interfaces
cannot be joined faithfully as they stand, and forcing them would build a
mechanism that turns an AWS outage into a fraud verdict.

### The mismatch

| | NAT | gateway operator signer |
|---|---|---|
| shape | `fn sign(&self, msg: &[u8]) -> Vec<u8>` | `async fn sign_hash(&self, &[u8;32]) -> Result<RecoverableSignature, X402Error>` |
| sync/async | sync | **async** |
| fallible | **no** | yes |

`SignedContribution::create(...) -> Self` is infallible too, so there is nowhere
for a signing error to go. An adapter has exactly two options when KMS is
unreachable: **panic** (kills the training loop) or **return garbage bytes**.

### Why garbage bytes is the worse one

`gather_and_aggregate` rejects an unverifiable contribution as
`RejectReason::BadSignature`, whose own doc reads *"forged, tampered, or unknown
node"*, and the contribution "contributes nothing to `total_reward_weight`".
There is **no transient-failure variant in the enum**. So a network blip at the
KMS endpoint would be recorded as the node forging signatures, and it would lose
that round's pay.

That is not a thing to ship and document. It is a thing to fix upstream.

### The ask (small, and it is NAT's to make)

```rust
pub trait Signer {
    fn node_id(&self) -> &str;
    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, SignError>;   // was -> Vec<u8>
}
```
and `SignedContribution::create(...) -> Result<Self, SignError>`. Optionally an
async variant; a blocking bridge is acceptable for a LOCAL signer
(`EncryptedFileSigner` is local crypto — no network, so the only realistic
failure is a locked vault), but not for KMS.

Once fallible, the adapter is genuinely small, and the identity story is neat:
the gateway signer is **recoverable secp256k1**, so `node_id` can be the operator
address and the verifier recovers it from the signature — no roster to
distribute, which is strictly better than the keyed-hash stand-in.

### What I did NOT do

I did not write the adapter. It would have compiled, passed a happy-path test,
and converted an AWS outage into a slashing-adjacent fraud record the first time
KMS hiccupped.

---

## 2b. (superseded) The original framing

`nat-federated` ships `ToyKeyedSigner` (`sig = H(key || msg || key)`),
self-labelled *"TEST STAND-IN — not for production"*, with no public-key
separation. Gate 4's signed gather runs on it today.

**The replacement already exists and is merged.**
`citrate-inference-gateway/gateway/src/signer.rs` is the INFER-S1 WP-C operator
signer: a signer framework with nonce and spend-cap enforcement and an **AWS KMS
adapter**, with live-KMS, Anvil-e2e and encrypted-signer test suites
(`gateway/tests/infer_wpc_*.rs`). Its custody review is signed off.

So the work is: implement `nat_federated::Signer` / `Verifier` over that signer
rather than writing a new one. Not a new key-management story — the same one,
reused. That is the right shape anyway: one audited signer in the federation, not
two.

**Sequencing:** this blocks Gate 4 (federated proof), not the single-node
training backend. The backend can be finished and exercised on one node first.

---

## 3. Gap: GGUF export-only — NOT a co-op blocker

Worth restating precisely, because it reads worse than it is.

`nat-candle`'s `gguf.rs` is export-only, and its own doc is honest about the
limit: it produces a valid GGUF container of NAT's weights, but making it
*execute* in stock Ollama also requires conforming to a llama.cpp-recognised
architecture, which the zone-partitioned graph is not.

**The co-op does not need GGUF.** Resume and checkpointing run on
`varmap.save`/`load` → `model.safetensors`, which is exactly what
`job_artifacts.rs` verifies `model_start_hash` against. GGUF is for *external
distribution* — getting a trained NAT model into the llama.cpp ecosystem — which
is a product question, not a training-path one.

Reclassified: **not a gap in the backend; an open item in distribution.**

---

## 4. Third gap, found while wiring (new)

`nat_candle::AutoregLm` exposes no per-step gradient or weight accessor —
`varmap` is private and `train_minibatched` runs the whole loop internally. The
federated contribution unit is a per-zone *weight delta*, so the backend obtains
it by differencing checkpoints before and after a step.

That works today with **no change to NAT**, because `AutoregLm` names its
parameters with zone identity already (`zone_HP.wq`, `zone_SM.log_a`,
`score_PF`) — which is what `zone_delta.rs` in this PR exploits. It is I/O-heavy
(save → read → diff per step). A `varmap` accessor upstream in `nat-candle`
would make it cheap, and is the obvious optimisation once the path is proven.

Recording it as a known cost rather than pretending the save/diff is free.

---

## 5. Order of work

1. **Finish the single-node backend** — `nat` git dep behind a feature, the two
   arms wired to `ModelBackend`, `honors_job_spec()` → `true` (which it may
   legitimately be: artifacts are verified and commitments reproduce).
2. **Declare the grid per job** and teach the challenger to read it. Small, and
   it closes the last way an honest worker gets slashed.
3. **Port the operator signer** into `nat_federated::Signer` — unblocks Gate 4.
4. **GGUF execution**, if and when external distribution matters. Not on this
   critical path.

Nothing above needs a contract change, and nothing needs a new key-management
design.
