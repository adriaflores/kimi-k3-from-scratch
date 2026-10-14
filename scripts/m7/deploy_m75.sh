#!/usr/bin/env bash
# From repository root: M7_HOST=vast bash scripts/m7/deploy_m75.sh
# Only overlays *M7.5-specific* scripts/payload. Never overwrites M7.3/M7.2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HOST="${M7_HOST:-vast}"
bash scripts/m7/local_preflight_m75.sh
command -v ssh >/dev/null 2>&1 || { echo 'ERROR: ssh missing' >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo 'ERROR: tar missing' >&2; exit 1; }
files=(
  scripts/m7/launch_m75.sh
  scripts/m7/run_m75.sh
  scripts/m7/status_m75.sh
  experiments/m7/m75_gate
)
echo "Deploying M7.5 to $HOST:/workspace/kimi (small overlay, no checkpoint or dataset)."
# Stream archive over SSH. All paths are relative to the existing /workspace/kimi root.
tar -czf - "${files[@]}" | MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" ssh "$HOST" 'mkdir -p /workspace/kimi/scripts/m7 /workspace/kimi/experiments/m7 /workspace/kimi/artifacts/m7/m75 /workspace/kimi/runs/m75 && tar -xzf - -C /workspace/kimi'
echo 'PASS: M7.5 deployed. Next: ssh vast '\''bash /workspace/kimi/scripts/m7/launch_m75.sh'\'''
