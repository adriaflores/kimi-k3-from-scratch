#!/usr/bin/env bash
# User-side explicit start; no direct training outside remote guarded runner.
set -euo pipefail
HOST="${M8_HOST:-vast}"
PHASE="${1:-}"
if [[ "$PHASE" != drill && "$PHASE" != continuous ]]; then
  echo 'Usage: M82_START=YES_M8_DRILL bash scripts/m8/launch_m82.sh drill' >&2
  echo 'After M8.3 gate: M82_START=YES_M8_CONTINUOUS bash scripts/m8/launch_m82.sh continuous' >&2
  exit 4
fi
[[ "${M82_START:-}" == "YES_M8_${PHASE^^}" ]] || { echo 'Explicit M82_START authorization missing; NO-GO' >&2; exit 4; }
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh "$HOST" \
  "M82_START=YES_M8_${PHASE^^} bash /workspace/kimi/scripts/m8/run_m82.sh '$PHASE'"
