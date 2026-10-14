#!/usr/bin/env bash
# Git Bash: ONLY this script can request training; execute after remote preflight and tariff check.
set -euo pipefail
HOST="${M7_HOST:-vast}"
if [[ "${M73_START:-}" != 'YES_RUN_80M_FROM_SCRATCH' ]]; then
  echo 'To explicitly launch: M73_START=YES_RUN_80M_FROM_SCRATCH bash scripts/m7/launch_m73.sh' >&2
  exit 4
fi
ssh "$HOST" 'M73_START=YES_RUN_80M_FROM_SCRATCH bash /workspace/kimi/scripts/m7/run_m73.sh'
