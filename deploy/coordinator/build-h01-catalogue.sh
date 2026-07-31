#!/usr/bin/env bash
#
# Build the H-01 re-run: genesis checkpoints for every (rung, arm, seed), staged
# on the mirror, and the job catalogue that points at them.
#
#   ./build-h01-catalogue.sh <droplet-ip> <nat-dir> <out-dir>
#
# The schedule is option B from the 2026-07-30 decision: CONSTANT tokens per
# parameter, anchored at the budget the 64M rung already received, and 4 epochs
# rather than 8.
#
# Why that anchor. The old schedule ran a fixed 1,000,000 sequences at every
# rung, which is 229 tokens/param at 248K and 1.0 at 64M — so "does the gap widen
# with scale?" was confounded with "does the gap widen as training becomes less
# adequate?". Scaling the sequence count with parameters makes tokens/param
# constant, leaving scale as the only variable. Anchoring at 64M's existing
# budget (rather than at compute-optimal) keeps the top rung directly comparable
# to the prior result, so the re-run isolates the merge-floor change instead of
# tangling it with a schedule change.
#
# The honest cost of that anchor: every rung trains at 1.0 unique token per
# parameter, which is well under-trained. Both arms are matched, so the ablation
# stays valid — but it measures the low-data regime, and H-01 must not claim
# more. Reaching ~20 tokens/param would need 1.28B tokens for the top rung and
# corpus-v6 has 306.5M; the binding constraint is data, not compute.
set -euo pipefail

HOST="${1:?usage: build-h01-catalogue.sh <droplet-ip> <nat-dir> <out-dir>}"
NAT="${2:?}"
OUT="${3:?}"
REMOTE_ROOT=/var/lib/citrate-artifacts
DATASET_HASH=0x027bcde07e148979a854b6e1d042299190f482d1ba5f4c6b2f17afc84da8e2fc

# rung label | target params | tok/s (f32, measured on GB10) — the tok/s column
# is only used to size the lease, so a slow member is not timed out mid-run.
RUNGS=(
  "248K 250000 21025"
  "1M   1000000 20652"
  "2M   2000000 18337"
  "4M   4000000 14447"
  "8M   8000000 8626"
  "32M  32000000 2976"
  "64M  64000000 1683"
)
SEEDS=(1 2 3)
ARMS=(nat dense)

TOP_PARAMS=64074343     # the 64M rung, measured
TOP_SEQS=1000000        # what it actually received
EPOCHS=4
SEQ_LEN=64
VOCAB=16384
DTYPE=f32
BATCH=64

mkdir -p "$OUT"
JOBS="$OUT/jobs.json"
echo "[" > "$JOBS"
first=1

say() { printf '\n\033[1;36m--\033[0m %s\n' "$*"; }

for entry in "${RUNGS[@]}"; do
  read -r rung target tps <<< "$entry"
  # Sequences proportional to parameters, so tokens/param is constant.
  seqs=$(( TOP_SEQS * target / TOP_PARAMS )); [ "$seqs" -lt 1 ] && seqs=1
  # One "step" is an optimizer pass over `max_windows` windows.
  steps=$(( seqs * EPOCHS / BATCH )); [ "$steps" -lt 1 ] && steps=1
  # Lease with 3x headroom over the GB10 estimate: a member on slower hardware
  # must not have the job yanked out from under them mid-run.
  secs=$(( seqs * EPOCHS * SEQ_LEN / tps * 3 + 3600 ))

  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      id="h01-${rung}-${arm}-seed${seed}"
      gdir="$OUT/genesis/$id"
      say "$id  seqs=$seqs steps=$steps"

      hash=$(cd "$NAT" && cargo run --release -q -p nat-candle --features cuda \
        --example genesis_checkpoint -- \
        "$arm" "$target" "$seed" "$VOCAB" "$SEQ_LEN" "$DTYPE" "$gdir" 2>/dev/null)

      # Stage under the hash the worker will verify against.
      ssh -o StrictHostKeyChecking=accept-new "root@${HOST}" \
        "mkdir -p ${REMOTE_ROOT}/models/${hash}"
      rsync -a "$gdir/model.safetensors" "$gdir/sidecar.nat.json" \
        "root@${HOST}:${REMOTE_ROOT}/models/${hash}/"

      # The dense arm has no zones, so share sampling is disabled rather than
      # left on to record an empty vector at every step.
      if [ "$arm" = "dense" ]; then share_every=0; else share_every=50; fi

      [ $first -eq 0 ] && echo "," >> "$JOBS"; first=0
      cat >> "$JOBS" <<EOF
  {
    "id": "$id",
    "requires": "h01",
    "payload": {
      "task": "train",
      "model_start_hash": "$hash",
      "dataset_hash": "$DATASET_HASH",
      "commitment_grid": "q16",
      "epoch": 0,
      "steps": $steps,
      "worker_shard": 0,
      "batch_size": $BATCH,
      "learning_rate": 0.003,
      "max_windows": $BATCH,
      "shards_per_step": 8,
      "seed": $seed,
      "share_every": $share_every,
      "dead_zone_patience": 5
    },
    "lease_secs": $secs,
    "max_attempts": 3
  }
EOF
    done
  done
done

echo "]" >> "$JOBS"
uv run python3 -c "import json;d=json.load(open('$JOBS'));print(f'  {len(d)} jobs')" 2>/dev/null \
  || python3 -c "import json;d=json.load(open('$JOBS'));print(f'  {len(d)} jobs')"

ssh "root@${HOST}" "chmod -R a+rX ${REMOTE_ROOT}"
say "catalogue written to $JOBS"
