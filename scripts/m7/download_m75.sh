#!/usr/bin/env bash
# From repo root: M7_HOST=vast bash scripts/m7/download_m75.sh
# Downloads only a compact ZIP+SHA into results/m7/m75; no .pt or corpus files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HOST="${M7_HOST:-vast}"
RUN_ID="$(MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" ssh "$HOST" 'cat /workspace/kimi/runs/m75/current_run_id.txt')"
[[ "$RUN_ID" =~ ^m75_resume_gate_[0-9]{8}T[0-9]{6}Z_[0-9]+$ ]] || { echo "Unexpected remote run ID: $RUN_ID" >&2; exit 1; }
REMOTE="/workspace/kimi/artifacts/m7/m75/$RUN_ID"
FILE="${RUN_ID}_results.zip"
LOCAL="results/m7/m75/$RUN_ID"
mkdir -p "$LOCAL"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" scp "$HOST:$REMOTE/$FILE" "$LOCAL/$FILE"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" scp "$HOST:$REMOTE/$FILE.sha256" "$LOCAL/$FILE.sha256"
(cd "$LOCAL" && sha256sum -c "$FILE.sha256")
echo "PASS: $LOCAL/$FILE"
echo 'Share this ZIP in the chat for the M7.5 GO/NO-GO review.'
