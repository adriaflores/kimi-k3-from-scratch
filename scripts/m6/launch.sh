#!/usr/bin/env bash
# Remote. Starts background run without depending on SSH connection.
set -euo pipefail
command -v tmux >/dev/null || { echo 'tmux missing; install before launch.' >&2; exit 1; }
command -v timeout >/dev/null || { echo 'timeout missing' >&2; exit 1; }
if tmux has-session -t m6-experiment 2>/dev/null; then
  echo 'm6-experiment already running'; exit 1
fi
mkdir -p /workspace/kimi/artifacts
rm -f /workspace/kimi/artifacts/m6_latest_run_id.txt /workspace/kimi/artifacts/m6_latest_status.txt
# timeout is a *process* guard, not an instance billing watchdog.
tmux new-session -d -s m6-experiment 'timeout -s INT -k 30 1320 bash /workspace/kimi/scripts/m6/run.sh > /workspace/kimi/artifacts/m6_console.log 2>&1'
echo 'Started tmux session m6-experiment. Check with status.sh.'
