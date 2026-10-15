#!/usr/bin/env bash
# Run on the LOCAL computer, before enabling Vast. Never calls ssh or CUDA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v python >/dev/null || { echo 'NO-GO: Python not on PATH' >&2; exit 2; }
for sh in scripts/m8/*.sh; do bash -n "$sh"; done
python -m py_compile scripts/m8/m81.py scripts/m8/m82.py scripts/m8/m83.py \
    scripts/m8/probe_resume_m83.py scripts/m8/backup_client_m82.py \
    scripts/m8/verify_final_local_m83.py
bash scripts/m8/local_preflight_m81.sh
bash scripts/m8/local_preflight_m82.sh
python -m unittest discover -s tests/m8 -p 'test_m83.py' -v
printf '\nM8.3 LOCAL PASS. Neither SSH nor a production checkpoint or GPU was used.\n'
printf 'Remote H100 restore, first independent 4096 backup and GO are still PENDING.\n'
