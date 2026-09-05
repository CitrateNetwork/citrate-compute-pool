#!/usr/bin/env bash
# RM-Q RC-3a tripwire: the hosted Rust workflow must not mask format or
# clippy failures. Local CI is the interim enforcement path while Actions is
# unavailable, so it checks the hosted definition too.

set -euo pipefail

workflow=".github/workflows/ci.yml"
if [[ ! -f "$workflow" ]]; then
  echo "::error::missing hosted CI workflow: $workflow"
  exit 1
fi

if rg -n 'continue-on-error:[[:space:]]*true' "$workflow"; then
  echo "::error::hosted CI contains a masked failure path: $workflow"
  exit 1
fi

echo "PASS: compute-pool hosted Rust gates are fail-closed."
