#!/usr/bin/env bash
#
# Stage the real corpus and checkpoint into the mirror, in the exact layout
# `ArtifactStore` uses — which is what lets the mirror be a plain static file
# server rather than a service.
#
#   ./stage-artifacts.sh <droplet-ip> /path/to/nat
#
# Content addressing does the heavy lifting here. The remote path IS the hash, so
# a path's bytes can never legitimately change, which makes rsync's skip-if-present
# both safe and correct: a re-run costs a directory listing, not 2.4 GB.
set -euo pipefail

HOST="${1:?usage: stage-artifacts.sh <droplet-ip> <nat-dir>}"
NAT="${2:?usage: stage-artifacts.sh <droplet-ip> <nat-dir>}"
REMOTE_ROOT=/var/lib/citrate-artifacts

CORPUS="$NAT/corpus/values-spine/corpus-v6/c64a034b203c4b1cb8c74944b934c4c36783a77fd4e56d63b785b475d22433cb"
CKPT="$NAT/checkpoints-64m/nat-seed2"

[ -d "$CORPUS" ] || { echo "corpus not found at $CORPUS" >&2; exit 1; }
[ -d "$CKPT" ]   || { echo "checkpoint not found at $CKPT" >&2; exit 1; }

say() { printf '\n\033[1;34m--\033[0m %s\n' "$*"; }

# The hashes ARE the addresses, so they are computed from the bytes rather than
# hardcoded. A hardcoded hash that drifts from the file would produce a mirror
# that serves the right bytes at the wrong path — invisible until a worker's
# fetch 404s.
say "hashing (keccak256, the same function the worker verifies with)"
hashes=$(python3 - "$CKPT/model.safetensors" "$CORPUS/manifest.json" <<'PY'
import sys, pathlib
try:
    from Crypto.Hash import keccak
except ImportError:
    sys.exit("need pycryptodome: pip install pycryptodome")
for p in sys.argv[1:]:
    h = keccak.new(digest_bits=256)
    h.update(pathlib.Path(p).read_bytes())
    print("0x" + h.hexdigest())
PY
)
MODEL_HASH=$(echo "$hashes" | sed -n 1p)
DATASET_HASH=$(echo "$hashes" | sed -n 2p)
echo "  model:   $MODEL_HASH"
echo "  dataset: $DATASET_HASH"

say "creating the store layout"
ssh -o StrictHostKeyChecking=accept-new "root@${HOST}" \
  "mkdir -p ${REMOTE_ROOT}/models/${MODEL_HASH} ${REMOTE_ROOT}/datasets/${DATASET_HASH}"

say "checkpoint (123 MB)"
rsync -a --info=progress2 "$CKPT/model.safetensors" \
  "root@${HOST}:${REMOTE_ROOT}/models/${MODEL_HASH}/"

say "sidecar"
# Declares the shape and the commitment grid. NOT content-addressed — it
# describes the checkpoint rather than being part of it — which is exactly why
# the runner re-checks the declared grid against the payload after resolving.
# Values are the real 64M run's, from h01-64m-corpus-v6-2026-07-07.log.
ssh "root@${HOST}" "cat > ${REMOTE_ROOT}/models/${MODEL_HASH}/sidecar.nat.json" <<'EOF'
{
  "architecture": "zone-partitioned",
  "commitment_grid": "q16",
  "vocab": 16384,
  "d": 1183,
  "seq_len": 128,
  "dtype": "bf16",
  "zones": [{"id":"SM"},{"id":"CB"},{"id":"HP"},{"id":"PF"},{"id":"CX"}]
}
EOF

say "corpus manifest (80 MB)"
rsync -a --info=progress2 "$CORPUS/manifest.json" \
  "root@${HOST}:${REMOTE_ROOT}/datasets/${DATASET_HASH}/"

say "corpus shards (2.3 GB, 185,475 files — the slow part)"
# Every shard, not just the hot set: the mirror cannot know which shards a
# future job's stride will select, and a 404 mid-run is a declined job.
rsync -a --info=progress2 --include='shard_*.json' --exclude='*' \
  "$CORPUS/" "root@${HOST}:${REMOTE_ROOT}/datasets/${DATASET_HASH}/"

say "fixing permissions"
ssh "root@${HOST}" "chmod -R a+rX ${REMOTE_ROOT}"

say "verifying the mirror serves what the worker will ask for"
ssh "root@${HOST}" "
  set -e
  test -s ${REMOTE_ROOT}/models/${MODEL_HASH}/model.safetensors
  test -s ${REMOTE_ROOT}/models/${MODEL_HASH}/sidecar.nat.json
  test -s ${REMOTE_ROOT}/datasets/${DATASET_HASH}/manifest.json
  n=\$(ls ${REMOTE_ROOT}/datasets/${DATASET_HASH}/shard_*.json 2>/dev/null | wc -l)
  echo \"  shards staged: \$n\"
  [ \"\$n\" -gt 0 ]
  du -sh ${REMOTE_ROOT}"

cat <<EOF

  Staged. Use these in a job payload:

    "model_start_hash": "${MODEL_HASH}",
    "dataset_hash":     "${DATASET_HASH}",

  and point workers at:

    CITRATE_ARTIFACT_MIRROR=https://mirror.citrate.ai
EOF
