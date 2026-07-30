---
created: 2026-07-29
branch: fix/refuse-to-earn-on-placeholder-backend
author: Claude (Opus 5), directed by @SaulBuilds
status: grading — A-1..A-4 checked against source and live chain; A-2's scope is corrected
relates:
  - citrate-chain/handoffs/DGX_RESPONSE_CONSENSUS_AND_ALF_2026-07-29.md  (#139, graded A-1/A-2, left A-3/A-4)
---

# Track A — graded against source, not against file listings

#139 graded A-1 and A-2 and said plainly that A-3/A-4 were **not verified**. This
finishes that, and corrects A-2.

The correction matters because the previous grading checked **which files exist**.
Reading **what they compute** gives a different answer.

---

## A-2 — "wiring, not greenfield" is HALF right, and the missing half is the ML

**Confirmed:** the daemon is an event logger. `training-worker/src/bin/main.rs`
`dispatch_logs` classifies each log and emits `"event observed"`. It never
constructs a `Worker`. The state machine (`worker.rs`), a Candle backend, a
libp2p transport, an HTTP chain client and five integration tests all exist. The
binary's own header calls the remainder "edge glue".

**The part that was missed: no backend trains the job the chain describes.**

| what the chain specifies | what the code does with it |
|---|---|
| `model_start_hash` | `load_starting_weights` **returns the hash it was handed**. Fetching is deferred to S2/S3 — the trait doc says so. |
| `dataset_hash` | Decoded into `TrainingJobSpec`, then **consumed by nothing**. The only grep hits are the decoder and its tests. |
| the training data | `CandleBackend::synthetic_input(epoch, step, shard)` — arithmetic over the indices. No dataset is read. |
| the loss | `loss = sum(y)`, which the code itself labels "placeholder — real impl uses a target". |
| the weights | "Initialize weights as a deterministic ramp". |

`CandleBackend`'s own module doc is unambiguous: *"The point of this reference
impl isn't ML accuracy — it's to prove the trait wiring works end-to-end against
real candle tensors, so the Llama-3.2-1B work that lands on H100 hardware can
drop in via the same trait without surprises."*

That is an honest, well-built **trait-conformance harness**. It is not a trainer.

### Why this is a money bug, not a documentation gap

`ComputePoolTraining` is **deployed and live on 40204** at
`0x6eb7d4160ebfaf92cd9377114160da62c09f87bf` (14.7 KB of code, `nextJobId = 0` —
never used). It pays per epoch via `EpochPaymentReleased`. A worker earns by
committing a Merkle root of its step commitments, and **the protocol cannot tell
a root produced by real training from one produced by a harness** — both are just
hashes of gradient tensors.

So "wiring dispatch" as scoped would produce a daemon that reads a real job,
ignores the model and dataset it names, trains a randomly-initialised toy linear
layer on synthetic input, commits legitimate-looking roots, and collects real
SALT — indistinguishable on-chain from a worker that did the job.

**Gated in this branch.** `ModelBackend::honors_job_spec()` and
`ChainClient::is_live_settlement()`, both defaulting to the safe answer, and
`Worker::run` refuses when settlement is live and the backend cannot honour the
spec. Tests 179 → 183. This does not implement A-2; it makes A-2 safe to
implement.

**Revised A-2 scope:** event→dispatch wiring is genuinely thin. The real work is
a backend that loads `model_start_hash` (IPFS) and trains on `dataset_hash` —
which is S2/S3 in this repo's own plan, not a follow-up to S1.

---

## A-1 — CONFIRMED, and the port is clean

`pool-coordinator` is **4,597 lines** with **zero** occurrences of "training" or
`ComputePoolTraining`. It polls `ComputePool.ComputeRequested`. So A-1 is a port,
exactly as #139 said.

Worth noting the port inherits A-2's problem: a coordinator that dispatches
training jobs coordinates whatever the workers compute. It should not be pointed
at live settlement before a spec-honouring backend exists either — the same gate
covers it, because the workers it dispatches carry it.

---

## A-3 — NOT BLOCKED. The mechanism already exists

#139 did not inspect this. `ContributionAccounting.recordContribution` is:

```solidity
require(isRecorder[msg.sender] || msg.sender == governance(), "Not authorized");
```

There is already a **recorder allowlist**, so an automated hook does not need a
contract change — it needs an authorised recorder address and something worth
recording. `ContributionAccounting` is live on 40204 (`0xcdd24773…`, 4.8 KB).

`MAX_CONTRIBUTORS` is capped and the contract is **not upgradeable**, so raising
the cap means a redeploy. Worth knowing before the recorder starts admitting
contributors automatically.

**A-3 is small, and it is downstream of A-2**: recording contributions from
placeholder training would write fake patronage into a real ledger.

---

## A-4 — the brief's blocker is right; its repo location is not

**Correction:** the brief says `citrate-coop` "is NOT cloned locally". In fact
`Citrate-Labs/citrate-coop/` exists and is **tracked inside the parent monorepo**
— it has no `.git` of its own and is not a clone of `CitrateNetwork/citrate-coop`
(which is a separate remote, last updated 2026-07-28).

That matters operationally: work done in the local directory commits to
`Citrate-Labs`, not to the coop repo, and the two can silently diverge. Decide
which is canonical before writing code there.

The EIP-170 blocker on `CitrateCooperativeFactory` was not re-measured here.

---

## Recommended order

1. **A-3** — smallest, and the recorder allowlist means no contract change. But
   it is only meaningful once there is real work to record.
2. **A-2's real half** — a backend that loads `model_start_hash` and trains on
   `dataset_hash`. This is the critical path for "ALF real compute", and it is
   S2/S3-sized, not S1-sized.
3. **A-1** — the coordinator port, once workers do real work.
4. **A-4** — settle the repo question first, then the EIP-170 library extraction.

The gate in this branch is what makes 1–3 safe to do in any order.

---

## Verified vs. assumed

**Verified by reading source or querying chain:** every code claim above, with
file and line; `ComputePoolTraining` and `ContributionAccounting` both live on
40204 with the sizes and `nextJobId` shown; `pool-coordinator`'s line count and
its zero training references; the `isRecorder` allowlist; `citrate-coop` having
no `.git`; the full test suite at 183 passing with the gate in place.

**Assumed / not checked:** the EIP-170 byte count on `CitrateCooperativeFactory`
(the brief's 31,694 is carried forward unverified); whether the local
`citrate-coop` directory matches the remote; whether any Llama-3.2-1B backend
exists outside this repo.
