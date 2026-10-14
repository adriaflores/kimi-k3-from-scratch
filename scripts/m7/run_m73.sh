#!/usr/bin/env bash
# Remote, deliberately gated launch. Invoked by local launch_m73.sh only.
set -euo pipefail
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
ART="$ROOT/artifacts/m7/m73"
RUNS="$ROOT/runs/m73"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
if [[ "${M73_START:-}" != 'YES_RUN_80M_FROM_SCRATCH' ]]; then
  echo 'ERROR: Explicit M73_START authorization missing. Refusing to train.' >&2
  exit 4
fi
command -v tmux >/dev/null || { echo 'tmux missing' >&2; exit 4; }
mkdir -p "$ART" "$RUNS"
if [[ -e "$ART/current_run_id" ]] || tmux has-session -t m7-m73 2>/dev/null; then
  echo 'Existing M7.3 state/session; refusing to launch a second pilot.' >&2; exit 4
fi
ID="m73_pilot_80m_v1_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ART/$ID"
RUN="$RUNS/$ID"
test ! -e "$OUT" && test ! -e "$RUN" || { echo 'Run ID collision' >&2; exit 4; }
mkdir "$OUT"
echo "Preflight prior to new training $ID"
if ! "$PY" "$ROOT/scripts/m7/m73.py" preflight --output "$OUT/preflight.json" --run-dir "$RUN"; then
  echo "PREFLIGHT FAILED; diagnostics at $OUT/preflight.json; NO TRAINING STARTED." >&2
  exit 5
fi
printf '%s\n' "$ID" > "$ART/current_run_id.tmp"
mv "$ART/current_run_id.tmp" "$ART/current_run_id"
printf '{"run_id":"%s","run_dir":"%s","scratch":true,"resume":null}\n' "$ID" "$RUN" > "$OUT/current.json"
printf 'starting\n' > "$OUT/runner_state.txt"
tmux new-session -d -s m7-m73 "bash '$ROOT/scripts/m7/worker_m73.sh' '$ID'"
echo "LAUNCHED: $ID"
echo "Check status: ssh vast 'bash /workspace/kimi/scripts/m7/status_m73.sh'"
