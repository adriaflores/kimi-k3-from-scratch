#!/usr/bin/env bash
# Git Bash: overlay only; never includes data, checkpoints or secrets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HOST="${M8_HOST:-vast}"
bash scripts/m8/local_preflight_m82.sh
files=(scripts/m8/m82.py scripts/m8/backup_client_m82.py scripts/m8/run_m82.sh
       scripts/m8/launch_m82.sh scripts/m8/status_m82.sh scripts/m8/backup_client_m82.sh)
echo "Deploy M8.2 overlay to $HOST (scripts only; no training)."
tar -czf - "${files[@]}" | MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh "$HOST" \
  'mkdir -p /workspace/kimi/scripts/m8 && tar -xzf - -C /workspace/kimi'
LOCAL_SHA="$(sha256sum scripts/m8/m82.py | awk '{print $1}')"
REMOTE_SHA="$(MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh "$HOST" 'sha256sum /workspace/kimi/scripts/m8/m82.py' | awk '{print $1}')"
[[ "$LOCAL_SHA" == "$REMOTE_SHA" ]] || { echo 'Upload SHA mismatch; NO-GO' >&2; exit 2; }
echo 'DEPLOY PASS. No M8 training started.'
