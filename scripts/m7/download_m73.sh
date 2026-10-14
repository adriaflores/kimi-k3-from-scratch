#!/usr/bin/env bash
# Git Bash: download small evidence ZIP first; checkpoint separately, verify SHA.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST="${M7_HOST:-vast}"
BACKUP="${M73_BACKUP_DIR:-/c/datasets/kimi_k3_backups}"
OUT="${M73_RESULTS_DIR:-results/m7/m73}"
mkdir -p "$BACKUP" "$OUT"
ID=$(ssh "$HOST" 'cat /workspace/kimi/artifacts/m7/m73/current_run_id')
[[ "$ID" =~ ^m73_pilot_80m_v1_[0-9]{8}T[0-9]{6}Z$ ]] || { echo 'Bad run ID'; exit 3; }
REMOTE="/workspace/kimi/artifacts/m7/m73/$ID"
RUNDIR="/workspace/kimi/runs/m73/$ID"
if ! ssh "$HOST" "test -f '$REMOTE/m73_results.zip.sha256'"; then
  echo 'No packaged results; call status/collect first.' >&2; exit 3
fi
mkdir -p "$OUT/$ID"
scp "$HOST:$REMOTE/m73_results.zip" "$OUT/$ID/m73_results.zip"
scp "$HOST:$REMOTE/m73_results.zip.sha256" "$OUT/$ID/m73_results.zip.sha256"
(cd "$OUT/$ID" && sha256sum -c m73_results.zip.sha256)
if ! ssh "$HOST" "test -f '$REMOTE/final_receipt.json'"; then
  echo "DIAGNOSTIC ONLY: no verified final checkpoint. Return $OUT/$ID/m73_results.zip to chat."
  exit 0
fi
scp "$HOST:$REMOTE/final_receipt.json" "$OUT/$ID/final_receipt.json"
# Python local used only for safe JSON receipt parsing, not for torch or training.
PYTHON="${M73_LOCAL_PYTHON:-python}"
read -r REMOTE_SHA REMOTE_SIZE < <("$PYTHON" - "$OUT/$ID/final_receipt.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p['sha256'],p['bytes'])
PY
)
DEST="$BACKUP/${ID}_final_last.pt"
if [[ -f "$DEST" ]]; then
  echo "Existing local checkpoint: checking SHA before skipping transfer"
  existing=$(sha256sum "$DEST" | awk '{print $1}')
  if [[ "$existing" != "$REMOTE_SHA" ]]; then
    echo "LOCAL FILE COLLISION: $DEST exists but hash differs. Not overwriting." >&2; exit 5
  fi
else
  scp "$HOST:$RUNDIR/last.pt" "$DEST.partial"
  actual=$(sha256sum "$DEST.partial" | awk '{print $1}')
  if [[ "$actual" != "$REMOTE_SHA" ]]; then
    echo 'TRANSFER FAILED SHA CHECK: leaving .partial for forensic inspection.' >&2; exit 5
  fi
  mv "$DEST.partial" "$DEST"
fi
actual=$(sha256sum "$DEST" | awk '{print $1}')
size=$(wc -c < "$DEST" | tr -d '[:space:]')
[[ "$actual" == "$REMOTE_SHA" && "$size" == "$REMOTE_SIZE" ]] || { echo 'Final size/hash mismatch'; exit 5; }
"$PYTHON" - "$ID" "$DEST" "$REMOTE_SHA" "$REMOTE_SIZE" "$OUT/$ID/m73_download_receipt.json" <<'PY'
import json,sys,datetime
run,path,digest,size,target=sys.argv[1:]
with open(target,'w',encoding='utf-8') as f:
 json.dump({'run_id':run,'file':path,'sha256':digest,'bytes':int(size), 'verified':True,
            'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
 f.write('\n')
PY
echo "DOWNLOAD VERIFIED: $DEST ($size bytes)"
echo "RETURN TO CHAT: $OUT/$ID/m73_results.zip + $OUT/$ID/m73_download_receipt.json"
echo 'Do not destroy the instance/disk until receipt and M7.4 are reviewed.'
