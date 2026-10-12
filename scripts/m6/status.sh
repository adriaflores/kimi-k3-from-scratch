#!/usr/bin/env bash
# Remote. Shows current/previous run and result status.
set -euo pipefail
ART=/workspace/kimi/artifacts
if tmux has-session -t m6-experiment 2>/dev/null;then echo 'RUNNING';else echo 'NOT RUNNING';fi
for f in m6_latest_run_id.txt m6_latest_status.txt;do
  if [[ -f "$ART/$f" ]];then printf '%s: ' "$f";cat "$ART/$f";fi
done
if [[ -f "$ART/m6_console.log" ]];then
  echo '--- console ---';tail -n 12 "$ART/m6_console.log"
fi
if [[ -f "$ART/m6_latest_run_id.txt" ]];then
  RUNID="$(cat "$ART/m6_latest_run_id.txt")"
  LOG="/workspace/kimi/runs/m6_experiments/$RUNID/experiment.log"
  if [[ -f "$LOG" ]];then echo '--- experiment ---';tail -n 12 "$LOG";fi
fi
if [[ -f "$ART/m6_latest_results.zip" ]];then ls -lh "$ART/m6_latest_results.zip";fi
