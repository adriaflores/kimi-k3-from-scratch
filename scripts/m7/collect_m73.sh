#!/usr/bin/env bash
# Remote forensic collection if worker was interrupted; no training or overwrite.
set -euo pipefail
ROOT=/workspace/kimi
ART="$ROOT/artifacts/m7/m73"
ID=$(cat "$ART/current_run_id")
OUT="$ART/$ID"
RUN="$ROOT/runs/m73/$ID"
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
if tmux has-session -t m7-m73 2>/dev/null; then
  echo 'Worker still active; refusing to package a moving log.' >&2; exit 4
fi
if [[ ! -f "$OUT/summary.json" ]]; then
  "$PY" "$ROOT/scripts/m7/m73.py" summary --run-dir "$RUN" --log-dir "$OUT" || true
fi
if [[ ! -e "$OUT/m73_results.zip" ]]; then
  "$PY" "$ROOT/scripts/m7/m73.py" package --run-dir "$RUN" --log-dir "$OUT"
fi
echo "Collected diagnostic evidence for $ID"
