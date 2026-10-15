#!/usr/bin/env bash
# Manual, fail-closed post-incident recovery on Vast, never auto-starts training.
# Select a VERIFIED surviving M8 snapshot after M8.2 supervisor PAUSED_INCIDENT.
set -euo pipefail
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
STEP="${M83_RECOVERY_STEP:-}"
[[ "$STEP" =~ ^[0-9]+$ ]] || { echo 'Set M83_RECOVERY_STEP to snapshot global step' >&2; exit 2; }
[[ "$STEP" -ge 4096 && "$STEP" -lt 183106 && $((STEP % 4096)) -eq 0 ]] || { echo 'Invalid M8 periodic recovery step' >&2; exit 2; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
"$PY" -u "$ROOT/scripts/m8/m83.py" probe --step "$STEP"
"$PY" -u "$ROOT/scripts/m8/m83.py" approve --phase incident --step "$STEP"
"$PY" -u "$ROOT/scripts/m8/m83.py" verify-go
printf 'INCIDENT RECOVERY APPROVED. Inspect evidence and explicitly relaunch continuous with M8.2.\n'
