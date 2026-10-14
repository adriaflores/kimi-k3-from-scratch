#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

uv run python -m unittest discover -s tests/m7 -p 'test_m73*.py' -v
uv run python scripts/m7/m73.py preflight --local \
  --bundle experiments/m7/m73 --output results/m7/m73_local_preflight.json

printf '\nLOCAL PASS; no CUDA and no training performed.\n'