#!/usr/bin/env bash
# Run on your local PC when M8.4 has finished. Verifies off-Vast final backup,
# downloads only small evidence ZIP, checks exact hashes. NEVER stops Vast.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HOST="${M8_HOST:-vast}"
[[ -n "${M82_LOCAL_BACKUPS:-}" ]] || { echo 'Set M82_LOCAL_BACKUPS (native C:/... path on Windows)' >&2; exit 4; }
# Last checkpoint MUST be transferred and ACKed before any final packaging.
python scripts/m8/backup_client_m82.py --once --host "$HOST" --directory "$M82_LOCAL_BACKUPS"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ssh -o BatchMode=yes "$HOST" \
  'PY=/workspace/kimi/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python; "$PY" /workspace/kimi/scripts/m8/m83.py final-package'
OUT=results/m8/m83_final
mkdir -p "$OUT"
BASE=/workspace/kimi/artifacts/m8/m83
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' scp -o BatchMode=yes "$HOST:$BASE/m8_final_evidence.zip" "$OUT/m8_final_evidence.zip"
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' scp -o BatchMode=yes "$HOST:$BASE/m8_final_evidence.zip.sha256" "$OUT/m8_final_evidence.zip.sha256"
(cd "$OUT" && sha256sum -c m8_final_evidence.zip.sha256)
python scripts/m8/verify_final_local_m83.py --backup-directory "$M82_LOCAL_BACKUPS" \
  --evidence-zip "$OUT/m8_final_evidence.zip" --receipt "$OUT/final_local_verification.json"
echo 'FINAL OFFSITE FILE AND EVIDENCE VERIFIED. Manually check and then shut down Vast; never stop based only on remote ACK.'
