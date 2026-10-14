#!/usr/bin/env bash
# From Windows: ssh vast 'bash /workspace/kimi/scripts/m7/launch_m75.sh'
# Safe: launches ONLY load_checkpoint probe, NEVER train().
set -euo pipefail
BASE=/workspace/kimi
RUNS="$BASE/runs/m75"
CURRENT="$RUNS/current_run_id.txt"
KIT="$BASE/experiments/m7/m75_gate"
test -s "$KIT/scripts/probe_resume_m75.py" || { echo 'M7.5 kit not deployed; run deploy_m75.sh' >&2; exit 1; }
command -v tmux >/dev/null || { echo 'tmux not installed on Vast' >&2; exit 1; }
if tmux has-session -t kimi_m75_gate 2>/dev/null; then
  echo 'M7.5 gate already running. Use status_m75.sh.' >&2
  exit 1
fi
mkdir -p "$RUNS"
RUN_ID="m75_resume_gate_$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN="$RUNS/$RUN_ID"
mkdir -p "$RUN"
printf '%s\n' "$RUN_ID" > "$CURRENT"
# Detached, as in M7.3. Internal run script captures failures and archives evidence.
tmux new-session -d -s kimi_m75_gate "bash '$BASE/scripts/m7/run_m75.sh' '$RUN'"
echo "M7.5 started (read-only resume gate; NO optimizer steps): $RUN_ID"
echo "Status: ssh vast 'bash /workspace/kimi/scripts/m7/status_m75.sh'"
