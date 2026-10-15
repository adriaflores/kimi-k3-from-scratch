#!/usr/bin/env bash
# One complete M8 overlay deployment: M8.1 + fixed M8.2 + M8.3 + frozen inputs.
# Must be launched locally. No GPU access, data or weights transferred.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HOST="${M8_HOST:-vast}"
bash scripts/m8/local_preflight_m83.sh
files=(
  scripts/m8/m81.py scripts/m8/remote_preflight_m81.sh
  scripts/m8/m82.py scripts/m8/backup_client_m82.py
  scripts/m8/run_m82.sh scripts/m8/launch_m82.sh scripts/m8/status_m82.sh
  scripts/m8/backup_client_m82.sh
  scripts/m8/m83.py scripts/m8/probe_resume_m83.py
  scripts/m8/run_gate_m83.sh scripts/m8/recover_incident_m83.sh
  scripts/m8/verify_final_local_m83.py
  experiments/m7/m75_gate/recipe/m8_full_6b.yaml
  experiments/m8/m81/inputs/m75_resume_receipt.json
)
for f in "${files[@]}"; do test -s "$f" || { echo "NO-GO: missing $f" >&2; exit 2; }; done
printf 'Deploying M8.1, M8.2 and M8.3 code only to %s:/workspace/kimi\n' "$HOST"
tar -czf - "${files[@]}" | MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh -o BatchMode=yes "$HOST" \
  'mkdir -p /workspace/kimi && tar -xzf - -C /workspace/kimi'
# Verifies EVERY transferred file, not just a single script.
sha256sum "${files[@]}" | MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh -o BatchMode=yes "$HOST" \
  'cd /workspace/kimi && sha256sum -c -'
# Pre-existing immutable M7.3 manifest must remain on the instance. Do not
# fabricate it from GitHub (the original provenance was intentionally ignored).
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh -o BatchMode=yes "$HOST" \
  'test -s /workspace/kimi/experiments/m7/m75_gate/evidence/m73_source_manifest.json && echo "M7.3 SOURCE MANIFEST PRESENT"' \
  || { echo 'NO-GO: original M7.3 source manifest is absent on Vast; recover the original evidence before M8.1 remote gate.' >&2; exit 2; }
echo 'M8 ALL-CODE DEPLOY PASS. No training launched.'
