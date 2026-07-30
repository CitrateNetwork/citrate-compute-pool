# Deploying the coordinator and the artifact mirror

Target: **`citrate-alf-gateway`** (`167.172.132.125`) — already provisioned, already
running Caddy, and the ALF planset designates it for exactly these endpoints
(`01_SCOPE_OF_WORK.md`: *"alf-gateway on DO droplet … gather status"*).

```sh
./deploy/coordinator/deploy.sh 167.172.132.125
```

Idempotent. Re-running rebuilds, restarts, and skips artifacts already staged.

## Two things a human must do first

**1. Open SSH on the droplet.** Port 22 is currently filtered by a host firewall
(no DigitalOcean firewall is attached, so it is `ufw` on the box). From the
DigitalOcean web console → Droplets → citrate-alf-gateway → Access → Console:

```sh
ufw allow 22/tcp && ufw reload && ufw status
```

**2. Two DNS records, in Cloudflare.** `citrate.ai` is on Cloudflare, not DO, so
this cannot be scripted from here.

| type | name | content | proxy |
|---|---|---|---|
| A | `coordinator` | `167.172.132.125` | **DNS only (grey)** |
| A | `mirror` | `167.172.132.125` | **DNS only (grey)** |

**Grey cloud, not orange.** A proxied record breaks Caddy's HTTP-01 challenge so
no certificate is issued, and it would push 2.4 GB of artifact traffic through
the CDN. Every other Citrate service record is already DNS-only — `auth`,
`comms` and `rpc` all resolve straight to droplet IPs.

## What gets installed

| | |
|---|---|
| `/usr/local/bin/citrate-training-coordinator` | built **on** the droplet (x86_64; the dev box is aarch64) |
| `citrate-training-coordinator.service` | systemd, `Restart=always`, runs as `citrate` |
| `/var/lib/citrate-coordinator/` | `state.json` (the lease table) + `jobs.json` (the catalogue) |
| `/var/lib/citrate-artifacts/` | the mirror's content-addressed tree |
| Caddy | two site blocks **appended** to the existing Caddyfile |

## Design notes

**The coordinator holds no secrets.** Worker identity is recovered from each
request's signature, so there is no API key to issue, store or rotate, and
nothing on this box to leak. The systemd hardening protects the droplet from the
service, not the other way round.

**The coordinator binds loopback only.** Caddy terminates TLS and proxies. A
Caddy misconfiguration cannot accidentally publish an unproxied port.

**The mirror is not a service.** It is Caddy serving a directory. That is
possible because the remote layout is byte-for-byte the layout
`ArtifactStore` uses locally — which also means any member can serve their own
store directory and be a valid mirror with no software at all.

**The mirror being public is safe, and that is load-bearing rather than
incidental.** Every artifact a worker fetches is verified against a hash the job
named or a `provenance_root` the verified manifest committed to. A hostile mirror
cannot substitute a checkpoint, poison a corpus, or insert a shard the manifest
does not commit to — the worst it can do is fail to serve, and a worker that
cannot fetch declines the job. So this ships with no signatures and no key
distribution, because they would protect a property content addressing already
guarantees while adding a problem it does not have.

**The Caddyfile is appended to, never replaced.** This box serves other things,
and a prod Caddyfile is edited in place. `deploy.sh` backs it up, checks for an
existing block before appending, and runs `caddy validate` before reloading.

**Artifacts are staged whole, not just the hot set.** A worker fetches only the
shards its stride selects (~8 per job against 185,475), but the mirror cannot
predict which a future job will want, and a 404 mid-run is a declined job.

## After deploying

```sh
curl -s https://coordinator.citrate.ai/v1/status | jq .
# {"counts":{"pending":0,...},"settlement":"shadow"}

curl -sI https://mirror.citrate.ai/models/0x…/model.safetensors | head -1
```

Point a worker at it:

```sh
CITRATE_COORDINATOR_URL=https://coordinator.citrate.ai \
CITRATE_ARTIFACT_MIRROR=https://mirror.citrate.ai \
CITRATE_ARTIFACT_STORE=./artifacts \
CITRATE_PROBE_PATH=./probe.json \
CITRATE_TRAINING_KEYSTORE_PATH=… CITRATE_TRAINING_KEYSTORE_PASSPHRASE=… \
  citrate-coop-worker
```

`settlement: "shadow"` is the honest state and should stay that way until the
settlement tolerance is fixed from fleet divergence data. Nothing is paid.

## Adding jobs

`/var/lib/citrate-coordinator/jobs.json`, then restart. Jobs already recorded
keep their state; only unseen ids are added. There is deliberately no
job-authoring API — a fleet this size does not need one, and not having one means
there is no privileged write path to secure.

```json
[
  {
    "id": "h01-64m-nat-seed1",
    "requires": "h01",
    "payload": {
      "task": "train",
      "model_start_hash": "0x…",
      "dataset_hash": "0x…",
      "commitment_grid": "q16",
      "epoch": 0, "steps": 100, "worker_shard": 0,
      "batch_size": 4, "learning_rate": 0.0001,
      "max_windows": 64, "shards_per_step": 8, "seed": 2026
    },
    "lease_secs": 180000,
    "max_attempts": 3
  }
]
```

`model_start_hash` and `dataset_hash` are printed by `stage-artifacts.sh`.
