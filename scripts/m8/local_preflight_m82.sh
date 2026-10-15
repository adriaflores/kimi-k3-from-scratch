#!/usr/bin/env bash
# Windows Git Bash or Linux; only tests and static checks, no Vast access.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v python >/dev/null || { echo 'Python required on PATH' >&2; exit 2; }
for f in local_preflight_m82.sh deploy_m82.sh run_m82.sh launch_m82.sh status_m82.sh backup_client_m82.sh; do bash -n "scripts/m8/$f"; done
python -m unittest discover -s tests/m8 -p 'test_m82.py' -v
python -m py_compile scripts/m8/m82.py scripts/m8/backup_client_m82.py
printf '\nM8.2 LOCAL PASS. No SSH, GPU, training, or checkpoint was accessed.\n'
