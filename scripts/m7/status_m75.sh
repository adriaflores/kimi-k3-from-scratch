#!/usr/bin/env bash
# From Windows: ssh vast 'bash /workspace/kimi/scripts/m7/status_m75.sh'
set -euo pipefail
BASE=/workspace/kimi
CURRENT="$BASE/runs/m75/current_run_id.txt"
if [[ ! -s "$CURRENT" ]]; then
  echo 'NOT LAUNCHED: deploy and launch M7.5 first.'
  exit 0
fi
RUN_ID="$(cat "$CURRENT")"
RUN="$BASE/runs/m75/$RUN_ID"
ZIP="$BASE/artifacts/m7/m75/$RUN_ID/${RUN_ID}_results.zip"
printf 'Run: %s\n' "$RUN_ID"
if tmux has-session -t kimi_m75_gate 2>/dev/null; then
  echo 'Status: RUNNING (resume gate only; no training)'
elif [[ -s "$RUN/exit_code.txt" ]]; then
  read -r rc < "$RUN/exit_code.txt"
  if [[ "$rc" == 0 ]]; then echo 'Status: DONE / probe PASS'; else echo "Status: DONE / probe FAIL (exit $rc)"; fi
else
  echo 'Status: STARTING / UNKNOWN (no worker result yet)'
fi
if [[ -s "$RUN/m75_resume_receipt.json" ]]; then
  grep -E '"status"|"original_step"|"next_step"|"next_lr"|"remaining_updates"|"error"' "$RUN/m75_resume_receipt.json" | head -n 10 || true
fi
[[ ! -f "$ZIP" ]] || echo "Artifact: $ZIP"
if [[ -f "$RUN/m75_probe.log" ]]; then
  echo 'Last log lines:'
  tail -n 9 "$RUN/m75_probe.log"
fi
