#!/usr/bin/env bash
# Git Bash: deploy only ~KB of M7.3 files, NEVER the 12.63 GB corpus.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST="${M7_HOST:-vast}"
paths=(scripts/m7/m73.py scripts/m7/preflight_m73.sh scripts/m7/launch_m73.sh
       scripts/m7/run_m73.sh scripts/m7/worker_m73.sh scripts/m7/status_m73.sh
       scripts/m7/download_m73.sh scripts/m7/deploy_m73.sh scripts/m7/collect_m73.sh
       experiments/m7/m73/pilot_80m.yaml experiments/m7/m73/pilot_contract.json
       experiments/m7/m73/m6_baseline_train_config.json)
for p in "${paths[@]}"; do test -f "$p" || { echo "Missing $p" >&2; exit 1; }; done
# Small stdin tar, no extra temporary artifact in the remote data disk.
tar -czf - "${paths[@]}" | ssh "$HOST" 'mkdir -p /workspace/kimi && tar -xzf - -C /workspace/kimi'
local_hash=$(sha256sum scripts/m7/m73.py | awk '{print $1}')
remote_hash=$(ssh "$HOST" 'sha256sum /workspace/kimi/scripts/m7/m73.py' | awk '{print $1}')
test "$local_hash" = "$remote_hash" || { echo "Deploy hash mismatch!" >&2; exit 2; }
echo "DEPLOY PASS: m73.py SHA256=$remote_hash, no data transfer and no training."
