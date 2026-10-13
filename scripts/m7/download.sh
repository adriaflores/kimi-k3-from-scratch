#!/usr/bin/env bash
# Local Git Bash. Download and verify per-run result ZIP, then merge with local receipts.
set -euo pipefail
M7_HOST="${M7_HOST:-vast}"
[[ "$M7_HOST" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe SSH alias' >&2;exit 1; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
REMOTE=/workspace/kimi/artifacts/m7
STATUS="$(ssh "$M7_HOST" "cat '$REMOTE/latest_status.txt'")"
[[ "$STATUS" != running && "$STATUS" != package_failed ]] || { echo "Run not ready: $STATUS" >&2;exit 1; }
RUNID="$(ssh "$M7_HOST" "cat '$REMOTE/latest_run_id.txt'")"
[[ "$RUNID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe remote run ID' >&2;exit 1; }
OUTDIR="$ROOT/results/m7"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${RUNID}_results.zip"
[[ ! -e "$OUT" ]] || { echo "Already downloaded: $OUT" >&2;exit 1; }
EXPECTED="$(ssh "$M7_HOST" "cat '$REMOTE/latest_results.sha256'")"
[[ "$EXPECTED" =~ ^[a-f0-9]{64}$ ]] || { echo 'Invalid remote hash' >&2;exit 1; }
scp "$M7_HOST:$REMOTE/latest_results.zip" "$OUT.partial"
ACTUAL="$(sha256sum "$OUT.partial" | awk '{print $1}')"
[[ "$EXPECTED" == "$ACTUAL" ]] || { echo 'ZIP checksum mismatch; keep instance on' >&2;exit 1; }
mv "$OUT.partial" "$OUT"
"${M7_LOCAL_PYTHON:-python}" scripts/m7/merge_evidence.py --remote-zip "$OUT" --run-id "$RUNID"
echo "M7.1 results received (remote status: $STATUS); don't stop GPU until M7.2 is completed and its evidence saved."
