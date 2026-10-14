#!/usr/bin/env bash
# Remote tmux worker. Holds one run with all stdout outside the run-dir.
set -u
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
ID="${1:?run_id required}"
[[ "$ID" =~ ^m73_pilot_80m_v1_[0-9]{8}T[0-9]{6}Z$ ]] || exit 4
OUT="$ROOT/artifacts/m7/m73/$ID"
RUN="$ROOT/runs/m73/$ID"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
printf 'running\n' > "$OUT/runner_state.txt"
# GPU telemetry sampling every 10s (no invasive profiling).
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$OUT/gpu.csv" 2>&1 &
SMI_PID=$!
"$PY" -u "$ROOT/scripts/m7/m73.py" watch --run-dir "$RUN" --log-dir "$OUT" > "$OUT/watcher.log" 2>&1 &
WATCH_PID=$!
# No --resume and no --checkpoint override. Kimi checkpoint is run-dir/last.pt.
"$PY" -u "$ROOT/reference/scripts/train.py" \
    --data-dir "$ROOT/data/full" \
    --model-config "$ROOT/reference/configs/model/kimi_k3_300m.yaml" \
    --training-config "$ROOT/experiments/m7/current/pilot_80m.yaml" \
    --device cuda --run-dir "$RUN" > "$OUT/training.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$OUT/runner_exit_code.txt"
touch "$OUT/stop_watcher"
wait "$WATCH_PID"; WATCH_RC=$?
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true
printf '{"train_exit_code":%s,"watcher_exit_code":%s,"ended_at_utc":"%s"}\n' \
  "$RC" "$WATCH_RC" "$(date -u +%FT%TZ)" > "$OUT/runner_state.json"
"$PY" "$ROOT/scripts/m7/m73.py" summary --run-dir "$RUN" --log-dir "$OUT" > "$OUT/summary.log" 2>&1
SUM_RC=$?
if [[ "$RC" -eq 0 && "$SUM_RC" -eq 0 ]]; then
   printf 'training_complete_pending_external_backup\n' > "$OUT/runner_state.txt"
else
   printf 'failed\n' > "$OUT/runner_state.txt"
fi
"$PY" "$ROOT/scripts/m7/m73.py" package --run-dir "$RUN" --log-dir "$OUT" >> "$OUT/summary.log" 2>&1
PKG_RC=$?
echo "M73 train=$RC watcher=$WATCH_RC summary=$SUM_RC package=$PKG_RC" >> "$OUT/summary.log"
exit "$(( RC != 0 || SUM_RC != 0 || PKG_RC != 0 ))"
