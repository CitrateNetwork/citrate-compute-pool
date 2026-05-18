---
created: 2026-05-18T04:55:00Z
branch: main
author: monorepo-split
status: active
split-from-monorepo-at: b3ccd5c7
split-from-monorepo-tag: pre-split-v0.4.0
archived-monorepo: https://github.com/CitrateNetwork/citrate-monorepo-archive
agentile-archive: https://github.com/CitrateNetwork/citrate-agentile-archive
---

# citrate-compute-pool

Training-pool coordinator + worker for the Citrate compute marketplace.

## Crates

| Path | Crate | Role |
|---|---|---|
| `pool-coordinator/` | `citrate-pool-coordinator` | Coordinates worker assignments, attestation validation, and reward settlement |
| `training-worker/` | `citrate-training-worker` | Worker daemon that executes training jobs and reports back to the coordinator |

## Chain dependency

Neither crate has a Rust-level dependency on chain crates. They talk to the chain via JSON-RPC over HTTP at runtime. No SSH deploy key needed.

## Quick start

```bash
cargo build --release

# Coordinator
cargo run --release --bin citrate-pool-coordinator -- --rpc-url http://localhost:8545

# Worker
cargo run --release --bin citrate-training-worker -- --coordinator-url http://localhost:8080
```

## Repository context

Split from the Citrate monorepo on 2026-05-18 via `git filter-repo`, preserving 208 commits of per-file history.

- **Monorepo archive**: https://github.com/CitrateNetwork/citrate-monorepo-archive
- **Agentile archive**: https://github.com/CitrateNetwork/citrate-agentile-archive
- **Chain**: https://github.com/CitrateNetwork/citrate-chain
- **Gateway**: https://github.com/CitrateNetwork/citrate-inference-gateway

## License

[MIT](LICENSE).
