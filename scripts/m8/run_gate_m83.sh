#!/usr/bin/env bash
# Run ON Vast AFTER step 4096 drill has ended and local backup ACK is verified.
# This gate is read-only regarding weights/model; it never calls train().
set -euo pipefail
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
test -x "$PY" || { echo 'NO-GO: missing pinned H100 Python' >&2; exit 2; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
"$PY" -u "$ROOT/scripts/m8/m83.py" drill-ready
"$PY" -u "$ROOT/scripts/m8/m83.py" probe --step 4096
"$PY" -u "$ROOT/scripts/m8/m83.py" approve --step 4096 --phase drill
"$PY" -u "$ROOT/scripts/m8/m83.py" verify-go
printf 'M8.3 H100 RECOVERY PASS. Only explicit M82_START=YES_M8_CONTINUOUS can start final training.\n'
