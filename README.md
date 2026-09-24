# citrate-compute-pool

*Part of the **[Citrate Network](https://citrate.ai)** — own the means of computation. · [Docs](https://docs.citrate.ai) · [Run a node](https://citrate.ai/download) · [Contribute → free membership](https://github.com/CitrateNetwork/.github/blob/main/CONTRIBUTING.md)*

> The pool-side daemons for the Citrate compute marketplace (chain 40204) — an
> inference **pool coordinator** that dispatches jobs across member nodes and
> settles on chain, and a **training coordinator + workers** that lease and run
> federated / data-parallel training jobs.

## What it is

`citrate-compute-pool` is a Rust workspace of three crates that talk to the chain
over JSON-RPC (no Rust dependency on the chain crates):

- **`pool-coordinator`** (`citrate-pool-coordinator`) — coordinates inference
  worker assignments, validates attestations, and settles rewards through the
  `ComputePool` contract on chain 40204.
- **`training-coordinator`** (`citrate-training-coordinator`) — an HTTP job-lease
  server: hands out training jobs, tracks leases, and recovers each worker's
  signing address from its requests.
- **`training-worker`** — two daemons: `citrate-training-worker` (a chain-event
  daemon that watches `ComputePoolTraining` jobs, `training`/`pipeline` modes) and
  `citrate-coop-worker` (a member-facing daemon that registers with the training
  coordinator and polls for work, running passes via a pluggable model backend).

- Concept docs: https://docs.citrate.ai/compute · Training: https://docs.citrate.ai/training
- Depends on a [citrate-chain](https://github.com/CitrateNetwork/citrate-chain) RPC
  + the deployed `ComputePool` / `ComputePoolTraining` contracts.

## Prerequisites

```bash
# Rust (repo pins a toolchain via rust-toolchain.toml — rustup honors it)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# System packages (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y build-essential clang cmake pkg-config git curl

# Optional: NVIDIA CUDA toolkit for GPU training (candle CUDA backend).
# The workers run on CPU without it.
```

## Build from source

```bash
git clone https://github.com/CitrateNetwork/citrate-compute-pool
cd citrate-compute-pool

cargo build --release
# Binaries in target/release/:
#   citrate-pool-coordinator  citrate-training-coordinator
#   citrate-training-worker   citrate-coop-worker
cargo test --workspace
```

## Run locally

### Inference pool coordinator (chain-settled)

Env-driven. Minimum for a local run against a devnet chain:

```bash
CITRATE_POOL_RPC_URL=http://localhost:8545 \
CITRATE_POOL_CHAIN_ID=40204 \
CITRATE_POOL_CONTRACT=0x<ComputePool address> \
CITRATE_POOL_PRIVATE_KEY_HEX=<64-hex secp256k1 key>          `# testnet/devnet only` \
CITRATE_POOL_MEMBER_ENDPOINTS="0xmember1=http://127.0.0.1:7001" \
CITRATE_POOL_METRICS_ADDR=127.0.0.1:9100 \
./target/release/citrate-pool-coordinator
```

Verify: it logs `chain_id` at startup and, if `CITRATE_POOL_METRICS_ADDR` is set,
serves Prometheus metrics — `curl -s http://127.0.0.1:9100/metrics`.

### Training coordinator + worker (HTTP job leasing)

```bash
# 1. Coordinator — HTTP job-lease server, binds 127.0.0.1:8088 by default
CITRATE_COORDINATOR_BIND=127.0.0.1:8088 \
CITRATE_COORDINATOR_STATE=./coordinator-state.json \
CITRATE_COORDINATOR_JOBS=./jobs.json \
./target/release/citrate-training-coordinator

# 2. Member worker — registers with the coordinator and polls for work
CITRATE_COORDINATOR_URL=http://127.0.0.1:8088 \
CITRATE_TRAINING_PRIVATE_KEY_HEX=<64-hex secp256k1 key>       `# testnet/devnet only` \
CITRATE_ARTIFACT_STORE=./artifacts CITRATE_SCRATCH=./scratch \
./target/release/citrate-coop-worker
```

Verify the coordinator is up:

```bash
curl -s http://127.0.0.1:8088/           # coordinator HTTP surface responds
# the coordinator logs pending/leased/done job counts on start and on each lease
```

## Connect it locally  ← the differentiator

The coordinators' upstream is the **local chain RPC + deployed compute contracts**;
the training worker's upstream is the **training coordinator's HTTP endpoint**.

1. Start a devnet node from
   [citrate-chain](https://github.com/CitrateNetwork/citrate-chain):
   `./target/release/citrate devnet` → `http://localhost:8545`.
2. Deploy the compute contracts to it — `ComputePool` and, for training,
   `ComputePoolTraining` (from citrate-chain: `script/DeployComputePoolTraining.s.sol`
   and the marketplace deploy scripts). Record the addresses.
3. Point the pool coordinator at the chain (`CITRATE_POOL_RPC_URL`,
   `CITRATE_POOL_CONTRACT`); point the chain-event training worker at it
   (`CITRATE_WORKER_RPC_URL`, `CITRATE_WORKER_CONTRACT`).
4. For the HTTP training flow, run `citrate-training-coordinator` and point
   `citrate-coop-worker` at it via `CITRATE_COORDINATOR_URL`.
5. Fund the coordinator/worker operator keys with native SALT on the devnet so
   settlement transactions can pay gas.

> **Outbound RPC policy:** a plaintext-remote RPC is rejected at config load —
> use `https://` for remote hosts, or plain `http://` to a loopback host (the
> local-devnet posture).

See the full multi-repo bring-up: https://docs.citrate.ai/local-stack

## Configuration

**pool-coordinator** (env): `CITRATE_POOL_RPC_URL` (default `http://127.0.0.1:8545`),
`CITRATE_POOL_CHAIN_ID` (40204), `CITRATE_POOL_CONTRACT`, wallet via
`CITRATE_POOL_KEYSTORE_PATH` + `CITRATE_POOL_KEYSTORE_PASSPHRASE` (production) or
`CITRATE_POOL_PRIVATE_KEY_HEX` (testnet), `CITRATE_POOL_WALLET_ADDRESS`,
`CITRATE_POOL_MEMBER_ENDPOINTS` (`addr=url,…`), `CITRATE_POOL_METRICS_ADDR`,
`CITRATE_POOL_POLL_INTERVAL_SECS`, `CITRATE_POOL_FROM_BLOCK`.

**training-coordinator** (env): `CITRATE_COORDINATOR_BIND` (default `127.0.0.1:8088`),
`CITRATE_COORDINATOR_STATE` (default `./coordinator-state.json`),
`CITRATE_COORDINATOR_JOBS` (optional job catalogue).

**training-worker** (`citrate-training-worker`, chain-event): `CITRATE_WORKER_MODE`
(`training`|`pipeline`), `CITRATE_WORKER_CONTRACT`, `CITRATE_WORKER_JOB_ID`,
`CITRATE_WORKER_RPC_URL` (default `https://rpc.citrate.ai` — set to your local RPC),
`CITRATE_WORKER_CHAIN_ID` (40204), wallet via `CITRATE_TRAINING_KEYSTORE_PATH` +
`_PASSPHRASE` or `CITRATE_TRAINING_PRIVATE_KEY_HEX`, `CITRATE_WORKER_METRICS_ADDR`.
**coop-worker**: `CITRATE_COORDINATOR_URL`, `CITRATE_ARTIFACT_STORE`,
`CITRATE_SCRATCH`, `CITRATE_PROBE_PATH`.

> Operator keys load from a Web3 Secret Storage **V3** keystore; keystores below
> the modern PBKDF2 iteration floor load with a `warn` (backward-compat), not a
> failure.

## Links

- Docs: https://docs.citrate.ai/compute
- Depends on: [citrate-chain](https://github.com/CitrateNetwork/citrate-chain) ·
  Related: [citrate-node-agent](https://github.com/CitrateNetwork/citrate-node-agent) (single-node seller) ·
  [citrate-inference-gateway](https://github.com/CitrateNetwork/citrate-inference-gateway)
- Contributing (DCO): CONTRIBUTING.md · Security: SECURITY.md · License: LICENSE

## License

Source-available under the Business Source License 1.1 (see [`LICENSE`](LICENSE)); converts to Apache-2.0 on the Change Date stated in the license. This is the commercial application-layer / core tier of Citrate's open-core model; the infrastructure tier is Apache-2.0. Licensor: Citrate Inc.
