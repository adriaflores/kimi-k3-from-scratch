#!/usr/bin/env bash
# Local Git Bash. No Vast instance required. Read-only dataset verification.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DATA_DIR="${M7_DATA_DIR:-C:/datasets/fineweb_edu_6b}"
"${M7_LOCAL_PYTHON:-python}" scripts/m7/m71.py local-preflight --data-dir "$DATA_DIR"
echo 'M7.1 local preflight complete. Keep Vast stopped; prepare M7.2 before uploading.'
