#!/usr/bin/env bash
# Local Git Bash, GPU off. Verify actual M6 config recipe and M7.1 receipt.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"${M7_LOCAL_PYTHON:-python}" scripts/m7/m72_validate.py --mode local --out results/m7/m72/local_preflight.json
