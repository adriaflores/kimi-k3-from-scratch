#!/usr/bin/env bash
# Remote: mirrors M6 status.sh with a separate M7 namespace.
set -euo pipefail
ART=/workspace/kimi/artifacts/m7
if tmux has-session -t m7-setup 2>/dev/null;then echo RUNNING;else echo 'NOT RUNNING';fi
for f in latest_run_id.txt latest_status.txt;do
  if [[ -f "$ART/$f" ]];then printf '%s: ' "$f";cat "$ART/$f";fi
done
if [[ -f "$ART/console.log" ]];then echo '--- console ---';tail -n 12 "$ART/console.log";fi
if [[ -f "$ART/latest_run_id.txt" ]];then
  RUNID="$(cat "$ART/latest_run_id.txt")"
  LOG="/workspace/kimi/runs/m7_preparation/$RUNID/experiment.log"
  if [[ -f "$LOG" ]];then echo '--- acceptance ---';tail -n 12 "$LOG";fi
fi
if [[ -f "$ART/latest_results.zip" ]];then ls -lh "$ART/latest_results.zip";fi
