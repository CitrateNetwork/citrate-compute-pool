# Deploying the coordinator and the artifact mirror

Target: **`citrate-alf-gateway`** (`167.172.132.125`) — already provisioned, already
running Caddy, and the ALF planset designates it for exactly these endpoints
(`01_SCOPE_OF_WORK.md`: *"alf-gateway on DO droplet … gather status"*).

```sh
./deploy/coordinator/deploy.sh 167.172.132.125
```

Idempotent. Re-running rebuilds, restarts, and skips artifacts already staged.

## Status: DEPLOYED 2026-07-30

Live and verified end to end — a worker with an empty store staged 203 MB from
the mirror and trained the real 64M checkpoint, producing an `epoch_root`
identical to two prior local runs.

| | |
|---|---|
| coordinator | https://coordinator.citrate.ai/v1/status |
| mirror | https://mirror.citrate.ai |
| TLS | Let's Encrypt, both names |

**The corpus was already on the droplet.** `citrate-alf-gateway` has a 200 GiB
volume (`nat-corpus-backup`) carrying corpus-v6, byte-verified identical to the
local copy by sha256 on manifest and sampled shards. It is mounted at
`/mnt/nat-corpus` (in `/etc/fstab`, `nofail`) and **symlinked** into the store
layout rather than copied, so the mirror serves 2.4 GB that never had to be
uploaded and does not consume the 80 GB boot disk. Only the 123 MB checkpoint
was transferred.

If that volume is ever detached, the dataset symlink dangles and every fetch
404s — the coordinator would then decline jobs rather than serve wrong bytes,
which is the right failure, but the cause would not be obvious from the logs.

## Recovering SSH, if it is ever lost again

Port 22 was firewalled with no working web console. What worked, entirely from
the CLI:

```sh
doctl compute droplet-action snapshot 583045283 --snapshot-name pre-rebuild --wait
doctl compute droplet-action rebuild 583045283 --image ubuntu-24-04-x64 --wait
```

A rebuild restores a clean image (ufw inactive) and re-injects the droplet's
registered SSH keys, **keeps the same IP** so DNS needs no change, and **does
not touch attached volumes** — the corpus survived it. Snapshot first; that is
what makes it reversible. The host key changes, so clear the stale entry:
`ssh-keygen -R <ip>`.

## Prerequisites (both now satisfied — kept for a rebuild elsewhere)

**1. SSH reachable.** See the recovery section above if it is not.

**2. Two DNS records, in Cloudflare.** `citrate.ai` is on Cloudflare, not DO, so
this cannot be scripted from here. Both records now exist.

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

`lease_secs` is the job's hard deadline, not how long a silent machine holds it:
a lease lives 15 minutes (`LEASE_RENEW_WINDOW_SECS`) and the worker extends it
with a heartbeat every 4 minutes while it trains (PBA-L3b-001).

## Admission policy (PBA-L3b-001)

Registration is unauthenticated and keys are free, so the coordinator does not
trust a machine's self-reported tier for the ladder, and it limits how much one
party can hold:

| Control | Setting |
|---|---|
| Vetted addresses | `CITRATE_COORDINATOR_H01_WORKERS=0x…,0x…`: may take any tier they evidence (unset = nobody is vetted) |
| Identities per client address (IPv6 per /64) | `CITRATE_COORDINATOR_MAX_WORKERS_PER_SOURCE` (default 16; vouched machines are exempt) |
| New identities per hour, all sources | 60/h, burst 30 (compile-time); a registration refused by the per-address cap does not spend it |
| Concurrent leases per identity | 1 |
| No-show penalty | a lapsed lease costs the key a 1 h cool-down, doubling per consecutive no-show up to 7 days; a delivered result from a vetted worker resets it. The history survives the key's record being evicted or its slot reclaimed |
| Source-group penalty | a lapse by an unvouched worker also (a) excludes its source group (IPv4 address, IPv6 /48) from that job and (b) puts the whole group on the same doubling cool-down, so a fresh key from the same network does not escape |
| Leases per source group | `CITRATE_COORDINATOR_MAX_LEASES_PER_SOURCE` (default 4; vouched machines are exempt) |
| Lapsed-job hold | after a lapse, the job goes only to vetted workers for 15 min. A job above the probe tier stays reserved beyond that while a vetted worker that could take it (claim satisfies the job, has not itself lapsed it) has asked for work or heartbeated within the last hour, for at most min(7 days, max(15 min, 2 × the job's `lease_secs`)) after the lapse. A registration alone does not count. Probe jobs return to the open pool after the 15 min |
| Work open to unvetted workers | `CITRATE_COORDINATOR_OPEN_TIER` = `probe` (default) or `federated`; H-01 is always vetted-only |
| Tier changes | a lease the current *operator policy* no longer allows is returned to the queue on its next heartbeat or submission, and at coordinator start. A worker re-registering with a different claimed capability keeps its lease until it completes or lapses |
| Signed-request limits | lease polls and heartbeats are limited per key (burst 10, then one every 2 s; 429 with `Retry-After` for that key only) and each is accepted once. Requests may be dated up to 2 min behind or 30 s ahead of the coordinator clock |
| Registration freshness | a registration carries a signed timestamp and must be within 2 min behind (30 s ahead) of the coordinator clock; an exact repeat is refused, also across a restart. A client without the timestamp gets 426 "worker too old". One key may re-register at most once a minute |
| Lease lifetime without a heartbeat | 15 min, extendable up to the job's `lease_secs` |

The client address is taken from `X-Forwarded-For` only when the TCP peer is
loopback, i.e. from Caddy, which overwrites that header with the real client
address (it trusts no inbound value unless `trusted_proxies` is set, and this
fragment does not set it). Do not add `trusted_proxies` without revisiting
`api::source_of`.

A network whose workers let a lease lapse is not offered that job again, and a
lapsed job goes to vetted workers first. `CITRATE_COORDINATOR_H01_WORKERS` is
the vetted list for every tier above the open one.

**Trade-off when `CITRATE_COORDINATOR_OPEN_TIER=federated`:** a federated job
that has lapsed once waits for a vetted worker whenever a vetted worker that
could take it is engaged, even if they are all busy on long jobs (for at most
min(7 days, max(15 min, 2 × the job's `lease_secs`)) after the lapse). That favours predictable
completion by known machines over throughput from the open pool. If no vetted
worker has been seen for an hour, the job returns to the open pool after the
15-minute hold. Operators who want more open-pool throughput should vet more
machines rather than shorten the hold.

State is written to disk only when something that must survive a restart
changes (a lease, heartbeat, submission, a change of claimed tier, an accepted
registration, expiry); a poll that finds no work, or a refused request, does
not rewrite it. Heartbeats are written in steps (when the expiry has moved 5
min past what is on disk, or reaches the deadline), and on start every lease
still inside its deadline gets a fresh 15-min window, so a restart never
charges a worker for the coordinator's own downtime. A failed write is retried
on the next request. Writes run off
the request-serving threads.

**Deploy order for this version: coordinator first, or coordinator and workers
in lock-step.**

- A new worker checks that the coordinator registered its own address and
  fails loudly if not; against an older coordinator (which does not know the
  timestamped registration format) it would otherwise register under a
  different address and be refused every lease. So the coordinator must be
  upgraded no later than the workers.
- Workers already registered with an older client keep leasing, heartbeating
  and submitting against the new coordinator until they restart; on restart
  their registration is refused with 426 "worker too old" until they are
  upgraded.
- If any worker in the fleet predates lease heartbeats, upgrade coordinator and
  workers in lock-step: such workers lose long jobs after the 15-minute
  renewal window.
- Set `CITRATE_COORDINATOR_H01_WORKERS` in the unit to the fleet's vetted
  worker addresses.
