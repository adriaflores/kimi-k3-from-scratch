#!/usr/bin/env bash
# From repository root on Windows Git Bash: bash scripts/m7/local_preflight_m75.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v python >/dev/null 2>&1 || { echo 'ERROR: python not on PATH' >&2; exit 1; }
python experiments/m7/m75_gate/scripts/static_gate_m75.py
for script in local_preflight_m75.sh deploy_m75.sh launch_m75.sh run_m75.sh status_m75.sh download_m75.sh; do
  bash -n "scripts/m7/$script"
done
test -s experiments/m7/m75_gate/scripts/probe_resume_m75.py
printf 'PASS: M7.5 overlay ready in repo root. No weights, data or secrets are bundled.\n'
