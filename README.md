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

## Keystore cipher hardening (ENCRYPT-S1 WP-10)

Both crates load an operator signing key from a Web3 Secret Storage **V3**
keystore (`CITRATE_POOL_KEYSTORE_PATH` / `CITRATE_TRAINING_WORKER_KEYSTORE_PATH`).
The V3 parameters we read — **PBKDF2-HMAC-SHA256 + AES-128-CTR** — are functional
and interoperable with the JS SDK (W-01) but weaker than the scrypt + AES-256
family used elsewhere.

**Plan (deferred to next key rotation, inventory A21):** migrate *newly minted*
keystores to **scrypt (N≥2¹⁸, r=8, p=1) + AES-256-CTR**. Minting is creator-side
(the JS SDK), not the Rust readers here.

**Disposition:** the Rust wallet readers (`pool-coordinator/src/wallet.rs`,
`training-worker/src/wallet.rs`) must keep decrypting the legacy
pbkdf2/aes-128-ctr format unchanged for backward compat — the cipher/KDF guards
stay permissive until every operator key is re-minted. As a non-breaking interim
signal, loading a keystore whose PBKDF2 iteration count is below the modern floor
(600 000, OWASP 2023) emits a `warn` log without failing the load. The
iteration-count bump for new keystores rides the scrypt migration above; it is
not applied in the readers because they have no creation path.

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
