#!/usr/bin/env bash
# Run in local Git Bash. Safe to call after the remote run has finished, including failed runs.
set -euo pipefail
M6_HOST="${M6_HOST:-vast-kimi}"
ROOT="$(git rev-parse --show-toplevel)"
REMOTE=/workspace/kimi/artifacts
STATUS="$(ssh "$M6_HOST" "cat '$REMOTE/m6_latest_status.txt'")"
[[ "$STATUS" != running && "$STATUS" != package_failed ]] || { echo "Run not ready: $STATUS" >&2; exit 1; }
RUNID="$(ssh "$M6_HOST" "cat '$REMOTE/m6_latest_run_id.txt'")"
[[ "$RUNID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe remote run ID' >&2; exit 1; }
OUTDIR="$ROOT/results/m6"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${RUNID}_results.zip"
[[ ! -e "$OUT" ]] || { echo "Already downloaded: $OUT" >&2; exit 1; }
EXPECTED="$(ssh "$M6_HOST" "cat '$REMOTE/m6_latest_results.sha256'")"
scp "$M6_HOST:$REMOTE/m6_latest_results.zip" "$OUT.partial"
ACTUAL="$(sha256sum "$OUT.partial" | awk '{print $1}')"
if [[ "$EXPECTED" != "$ACTUAL" ]];then
  echo "SHA256 mismatch! Expected $EXPECTED, got $ACTUAL" >&2; exit 1
fi
mv "$OUT.partial" "$OUT"
echo "Downloaded & SHA256 verified: $OUT (status: $STATUS)"
echo 'STOP/PAUSE Vast instance manually; confirm status and review remaining storage charges.'
