#!/usr/bin/env bash
# Git Bash / Linux: bash scripts/m8/local_preflight_m81.sh
# No SSH, CUDA, checkpoint reading or training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python -m unittest discover -s tests/m8 -p 'test_m81.py' -v
python scripts/m8/m81.py --mode local --receipt results/m8/m81_local_preflight.json
bash -n scripts/m8/local_preflight_m81.sh
bash -n scripts/m8/remote_preflight_m81.sh
printf '\nM8.1 LOCAL PASS: remote preflight NOT performed; no training started.\n'
