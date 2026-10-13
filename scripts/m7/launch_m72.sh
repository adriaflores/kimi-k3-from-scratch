#!/usr/bin/env bash
# Remote Vast shell, after M7.1 success. NO TRAINING.
set -euo pipefail
ROOT=/workspace/kimi
ART="$ROOT/artifacts/m7/m72"
command -v tmux >/dev/null || { echo 'tmux missing' >&2; exit 1; }
command -v timeout >/dev/null || { echo 'timeout missing' >&2; exit 1; }
if tmux has-session -t m7-setup 2>/dev/null; then echo 'M7.1 still running' >&2; exit 1; fi
if tmux has-session -t m7-m72 2>/dev/null; then echo 'M7.2 already running' >&2; exit 1; fi
[[ -f "$ROOT/artifacts/m7/latest_status.txt" ]] && [[ "$(cat "$ROOT/artifacts/m7/latest_status.txt")" == success ]] || { echo 'M7.1 not accepted' >&2; exit 1; }
[[ -f "$ROOT/experiments/m7/current/pilot_80m.yaml" ]] || { echo 'M7.2 config missing; rerun M7.1 upload.sh' >&2; exit 1; }
mkdir -p "$ART"
rm -f "$ART/latest_run_id.txt" "$ART/latest_status.txt"
TIMEOUT_SECONDS="${M72_SETUP_TIMEOUT_SECONDS:-600}"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && (( TIMEOUT_SECONDS >= 60 && TIMEOUT_SECONDS <= 1800 )) || exit 2
tmux new-session -d -s m7-m72 "timeout -s INT -k 30 $TIMEOUT_SECONDS bash /workspace/kimi/scripts/m7/run_m72.sh > '$ART/console.log' 2>&1"
echo 'M7.2 contract gate launched in tmux m7-m72 (no training).'
