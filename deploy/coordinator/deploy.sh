#!/usr/bin/env bash
#
# Deploy the training coordinator + artifact mirror to citrate-alf-gateway.
#
#   ./deploy/coordinator/deploy.sh 167.172.132.125
#
# Idempotent: safe to re-run. It never resets the droplet's Caddyfile — a prod
# Caddyfile is edited, never replaced — and it never touches artifacts already
# staged (they are content-addressed, so re-uploading identical bytes is waste).
#
# Builds ON the droplet. The dev box is aarch64 and the droplet is x86_64, so
# cross-compiling would mean shipping a binary nobody has run on the target.
set -euo pipefail

HOST="${1:?usage: deploy.sh <droplet-ip>}"
SSH="ssh -o StrictHostKeyChecking=accept-new root@${HOST}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAT_DIR="${NAT_DIR:-$(cd "$REPO_ROOT/../nat" && pwd)}"

say() { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }

# ── 1. Preconditions, checked before anything is changed ─────────────────
say "checking the droplet, installing Caddy if absent"
$SSH 'set -e
  if ! command -v caddy >/dev/null; then
    echo "  installing caddy"
    apt-get update -qq
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
      | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    apt-get update -qq && apt-get install -y -qq caddy
  fi
  free_kb=$(df --output=avail / | tail -1)
  [ "$free_kb" -gt 8000000 ] || { echo "less than 8 GB free on /"; exit 1; }
  echo "  ok: caddy present, $(( free_kb / 1024 / 1024 )) GB free, $(uname -m)"'

# ── 2. Users and directories ─────────────────────────────────────────────
say "creating the service user and directories"
$SSH 'set -e
  id -u citrate >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin citrate
  mkdir -p /var/lib/citrate-coordinator /var/lib/citrate-artifacts /var/log/caddy
  chown -R citrate:citrate /var/lib/citrate-coordinator
  # Caddy runs as `caddy`, not root. Without this its log writer cannot open the
  # file and the whole config reload fails — while `systemctl reload` still
  # exits 0 and Caddy keeps serving the previous config. Cost an hour once.
  chown -R caddy:caddy /var/log/caddy
  # The mirror is read-only to the world and writable only by root (rsync target).
  chmod 755 /var/lib/citrate-artifacts'

# ── 3. Toolchain + build, on the target ──────────────────────────────────
say "installing the rust toolchain on the droplet (first run only)"
$SSH 'set -e
  if ! su - root -c "command -v cargo" >/dev/null 2>&1 && [ ! -x /root/.cargo/bin/cargo ]; then
    apt-get update -qq && apt-get install -y -qq build-essential pkg-config libssl-dev git curl
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.96.0 --profile minimal
  fi
  echo "  cargo: $(/root/.cargo/bin/cargo --version)"'

say "shipping the source"
# Source only — no target/, no artifacts. Those are handled separately.
rsync -az --delete \
  --exclude target/ --exclude .git/ --exclude deploy/ \
  "$REPO_ROOT/training-coordinator" "$REPO_ROOT/training-worker" \
  "$REPO_ROOT/pool-coordinator" "$REPO_ROOT/Cargo.toml" "$REPO_ROOT/Cargo.lock" \
  "$REPO_ROOT/rust-toolchain.toml" "$REPO_ROOT/.cargo" \
  "root@${HOST}:/opt/citrate-compute-pool/"

say "building on the droplet (x86_64 — this takes a few minutes the first time)"
# Only the coordinator. The worker's `nat` feature pulls candle and is not
# needed here: this box hands out work, it does not train.
#
# The private deps are fetched through a FORWARDED ssh agent (-A), so no deploy
# key or token is ever written to the droplet — the credential lives for exactly
# the length of this command and dies with the connection. The `insteadOf`
# rewrite is what lets an https:// dependency URL travel over that agent.
ssh -A -o StrictHostKeyChecking=accept-new "root@${HOST}" 'set -e
  git config --global url."ssh://git@github.com/".insteadOf "https://github.com/"
  ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null || true
  sort -u /root/.ssh/known_hosts -o /root/.ssh/known_hosts 2>/dev/null || true
  cd /opt/citrate-compute-pool
  export PATH=/root/.cargo/bin:$PATH
  export CARGO_NET_GIT_FETCH_WITH_CLI=true
  cargo build --release -p citrate-training-coordinator
  install -m 0755 target/release/citrate-training-coordinator /usr/local/bin/
  echo "  installed: $(/usr/local/bin/citrate-training-coordinator --help 2>&1 | head -1 || echo built)"'

# ── 4. systemd ───────────────────────────────────────────────────────────
say "installing the systemd unit"
scp -q "$REPO_ROOT/deploy/coordinator/citrate-training-coordinator.service" \
  "root@${HOST}:/etc/systemd/system/"
$SSH 'set -e
  # Seed an empty catalogue on first deploy; never clobber a live one.
  [ -f /var/lib/citrate-coordinator/jobs.json ] || echo "[]" > /var/lib/citrate-coordinator/jobs.json
  chown citrate:citrate /var/lib/citrate-coordinator/jobs.json
  systemctl daemon-reload
  systemctl enable --now citrate-training-coordinator
  sleep 2
  systemctl is-active --quiet citrate-training-coordinator && echo "  active" || {
    journalctl -u citrate-training-coordinator -n 30 --no-pager; exit 1; }'

# ── 5. Caddy — append, never replace ─────────────────────────────────────
say "wiring Caddy (appending; the existing Caddyfile is preserved)"
scp -q "$REPO_ROOT/deploy/coordinator/Caddyfile.fragment" "root@${HOST}:/tmp/citrate-coordinator.caddy"
$SSH 'set -e
  cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak.$(date +%s)"
  if grep -q "coordinator.citrate.ai" /etc/caddy/Caddyfile; then
    echo "  already present — leaving the Caddyfile alone"
  else
    printf "\n" >> /etc/caddy/Caddyfile
    cat /tmp/citrate-coordinator.caddy >> /etc/caddy/Caddyfile
    echo "  appended"
  fi
  # `caddy validate` checks SYNTAX only. It does not test whether the running
  # process can actually apply the config — not file permissions, not ports.
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
  systemctl reload caddy
  # And `systemctl reload` exits 0 even when Caddy rejects the new config over
  # its admin API and carries on serving the OLD one. That failure is entirely
  # silent, so the reload is verified rather than assumed: no 443 listener means
  # the config did not take.
  sleep 3
  if ! ss -tln | grep -q ":443 "; then
    echo "  caddy did not bind 443 after reload — the config was rejected:"
    journalctl -u caddy -n 20 --no-pager | grep -i error | tail -5
    exit 1
  fi
  echo "  reload verified (443 bound)"'

# ── 6. Artifacts ─────────────────────────────────────────────────────────
# rsync skips bytes already present, so a re-run costs a directory listing
# rather than 2.4 GB. Content addressing makes that safe: a path's bytes can
# never legitimately change, because the path IS the hash.
say "staging artifacts (2.5 GB on the first run; near-zero afterwards)"
if [ -n "${SKIP_ARTIFACTS:-}" ]; then
  echo "  SKIP_ARTIFACTS set — skipping"
else
  "$REPO_ROOT/deploy/coordinator/stage-artifacts.sh" "$HOST" "$NAT_DIR"
fi

say "done"
cat <<EOF

  coordinator  https://coordinator.citrate.ai/v1/status
  mirror       https://mirror.citrate.ai/

  Both need DNS A records pointing at ${HOST}, DNS-only (grey cloud) in
  Cloudflare — a proxied record blocks Caddy's HTTP-01 challenge and would put
  the 2.4 GB mirror through the CDN.

  Verify:
    curl -s https://coordinator.citrate.ai/v1/status | jq .
    ssh root@${HOST} journalctl -u citrate-training-coordinator -f
EOF
