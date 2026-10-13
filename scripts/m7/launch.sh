#!/usr/bin/env bash
# Remote: background M7.1 acceptance checks, disconnected from SSH.
set -euo pipefail
command -v tmux >/dev/null || { echo 'tmux missing' >&2;exit 1; }
command -v timeout >/dev/null || { echo 'timeout missing' >&2;exit 1; }
if tmux has-session -t m7-setup 2>/dev/null;then echo 'm7-setup already running' >&2;exit 1;fi
ROOT=/workspace/kimi
[[ -f "$ROOT/experiments/m7/current/run.py" ]] || { echo 'M7 experiment missing' >&2; exit 1; }
mkdir -p "$ROOT/artifacts/m7"
rm -f "$ROOT/artifacts/m7/latest_run_id.txt" "$ROOT/artifacts/m7/latest_status.txt"
# Process guard, never stops Vast instance. Enough to hash 12.6 GB on host.
TIMEOUT_SECONDS="${M7_SETUP_TIMEOUT_SECONDS:-900}"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || exit 2
tmux new-session -d -s m7-setup "timeout -s INT -k 30 $TIMEOUT_SECONDS bash /workspace/kimi/scripts/m7/run.sh > /workspace/kimi/artifacts/m7/console.log 2>&1"
echo 'Started tmux m7-setup; check scripts/m7/status.sh.'
