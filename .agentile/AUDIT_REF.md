---
created: 2026-05-31T18:00:00Z
branch: audit/audit-ref-2026-05-31
author: claude-opus-4-8 (audit wiring)
status: active
audit_id: 2026-05-31-federation-deep-audit
---

# Active audit reference — `citrate-compute-pool`

> This repo's two-way link into the centralized federation audit trail
> (Hybrid topology). It is what an outsider cloning *this* repo follows to the
> findings. The canonical audit home is the `citrate-security` repo.

This repo participates in the **Inaugural Federation Deep Audit** opened
2026-05-31 as a **Tier 1 — full audit** surface.

- Audit root: `citrate-security/audits/2026-05-31-federation-deep-audit/`
- This repo's folder: `.../per-repo/citrate-compute-pool/` — `MAP.md` (architecture + data-flow +
  two Mermaid diagrams) and `INVENTORY.md` (feature/app/function table)
- Findings roll-up: `.../06_FINDINGS.md`
- Audit index: `citrate-security/audits/AUDIT_INDEX.md`
- Standard: `citrate-security/.agentile/standard/AGENTILE_AUDIT_STANDARD.md`

## Local evidence (Hybrid topology)

Remediation diffs and post-fix e2e/regression evidence for findings against this
repo land **here**, next to the code, under
`.agentile/audits/2026-05-31-federation-deep-audit/` and are linked back from the
central finding.

## Scope for this repo

- Phase-1 mapping complete; see `per-repo/citrate-compute-pool/MAP.md` for the mapped surface and
  the high-risk areas queued for Phase-2 vuln-hunting.

## Federation-wide audit 2026-06-20 (FWA-C8) — remediation

- Audit root: `citrate-security/audits/2026-06-20-federation-wide-audit/per-chunk/FWA-C8/`
- Remediation log + tripwire (local, Hybrid topology):
  `.agentile/audits/2026-06-21-fwa-remediation/REMEDIATION_LOG.md`
  and `.../semgrep/fwa-c8-01-outbound-tls-gate.yaml`
- **FWA-C8-01** (MEDIUM, coordinator outbound plaintext `http://` with no TLS
  gate) — **FIXED** on branch `remediation/fwa-2026-06`. Ported node-agent's
  `validate_outbound_url` into `pool-coordinator/src/outbound.rs`; applied at
  member-endpoint parse, RPC config-load, RPC adapter boundary, and the
  ws-subscription dial (sweep-found extra sink). RED→green, mutation 24/24 on
  the gate, semgrep tripwire clean on production source.
