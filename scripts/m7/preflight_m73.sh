#!/usr/bin/env bash
# Remote read-only preflight: GPU identification and dataset metadata, NO model run.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
test -x "$PY" || { echo "Pinned M6 Python missing: $PY" >&2; exit 1; }
OUT="$ROOT/artifacts/m7/m73/preflight_only"
mkdir -p "$OUT"
"$PY" "$ROOT/scripts/m7/m73.py" preflight --output "$OUT/preflight.json"
echo "REMOTE PREFLIGHT PASS: $OUT/preflight.json (no training)."
