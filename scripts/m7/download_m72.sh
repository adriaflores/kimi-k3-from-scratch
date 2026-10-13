#!/usr/bin/env bash
# Local Git Bash: verified remote evidence + local recipe receipt.
set -euo pipefail
M7_HOST="${M7_HOST:-vast}"
[[ "$M7_HOST" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe SSH alias' >&2;exit 1; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
REMOTE=/workspace/kimi/artifacts/m7/m72
STATUS="$(ssh "$M7_HOST" "cat '$REMOTE/latest_status.txt'")"
[[ "$STATUS" == success || "$STATUS" == failed_* ]] || { echo "M7.2 not finished: $STATUS" >&2;exit 1; }
RUNID="$(ssh "$M7_HOST" "cat '$REMOTE/latest_run_id.txt'")"
[[ "$RUNID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe remote ID' >&2;exit 1; }
mkdir -p results/m7
OUT="results/m7/${RUNID}_results.zip"
[[ ! -e "$OUT" ]] || { echo "Refuse overwrite: $OUT" >&2;exit 1; }
EXPECTED="$(ssh "$M7_HOST" "cat '$REMOTE/latest_results.sha256'")"
[[ "$EXPECTED" =~ ^[a-f0-9]{64}$ ]] || { echo 'Invalid remote SHA' >&2;exit 1; }
scp "$M7_HOST:$REMOTE/latest_results.zip" "$OUT.partial"
ACTUAL="$(sha256sum "$OUT.partial" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'ZIP checksum mismatch' >&2;exit 1; }
mv "$OUT.partial" "$OUT"
if [[ "$STATUS" != success ]];then
  echo "M7.2 FAILED ($STATUS); diagnostic results ZIP: $OUT"
  exit 2
fi
"${M7_LOCAL_PYTHON:-python}" scripts/m7/m72_merge_evidence.py --remote-zip "$OUT" --run-id "$RUNID"
echo 'M7.2 evidence received and verified. Keep Vast on until both M7.1 and M7.2 are backed up.'
