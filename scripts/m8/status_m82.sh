#!/usr/bin/env bash
# Remote metadata-only status. No GPU or checkpoint reads.
set -euo pipefail
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
if tmux has-session -t kimi_m82 2>/dev/null; then echo 'TMUX: active'; else echo 'TMUX: inactive'; fi
"$PY" "$ROOT/scripts/m8/m82.py" status
