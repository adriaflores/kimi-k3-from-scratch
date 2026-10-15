#!/usr/bin/env bash
# For the future single Vast activation, AFTER M8.1-3 local validation.
# Example: M8_PRICE_EUR_H=2.3 M8_COST_CAP_EUR=100 bash /workspace/kimi/scripts/m8/remote_preflight_m81.sh
# Does NOT launch training and does NOT deploy any code.
set -euo pipefail
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
test -x "$PY" || { echo "Missing pinned M7.3 Python: $PY" >&2; exit 2; }
: "${M8_PRICE_EUR_H:?Set confirmed current hourly price converted to EUR (not historical)}"
: "${M8_COST_CAP_EUR:?Set explicit EUR budget cap}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
exec "$PY" "$ROOT/scripts/m8/m81.py" --mode remote \
  --price-per-hour-eur "$M8_PRICE_EUR_H" --cost-cap-eur "$M8_COST_CAP_EUR" \
  --projected-hours "${M8_PROJECTED_HOURS:-30}" \
  --min-free-gib "${M8_MIN_FREE_GIB:-20}" \
  --proposed-run-dir "${M8_PROPOSED_RUN_DIR:-/workspace/kimi/runs/m8/m8_full_6b_v1}" \
  --source-manifest "${M8_SOURCE_MANIFEST:-/workspace/kimi/experiments/m7/m75_gate/evidence/m73_source_manifest.json}" \
  --receipt /workspace/kimi/artifacts/m8/m81/preflight.json
