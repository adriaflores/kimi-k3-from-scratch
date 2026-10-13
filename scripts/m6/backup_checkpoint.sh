#!/usr/bin/env bash
# Local Git Bash. Optional but recommended for full M6 external checkpoint proof.
# After download.sh, before pausing Vast. Copies original initial checkpoint.
set -euo pipefail
M6_HOST="${M6_HOST:-vast-kimi}"
ROOT="$(git rev-parse --show-toplevel)"
RUNID="$(ssh "$M6_HOST" 'cat /workspace/kimi/artifacts/m6_latest_run_id.txt')"
[[ "$RUNID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Unsafe remote run ID" >&2; exit 1; }
REMOTE="/workspace/kimi/runs/m6_experiments/$RUNID/data/initial/run/last.pt"
OUT_DIR="${M6_CHECKPOINT_BACKUP_DIR:-/c/datasets/kimi_k3_backups}"
STATUS="$(ssh "$M6_HOST" 'cat /workspace/kimi/artifacts/m6_latest_status.txt')"
[[ "$STATUS" == success ]] || { echo "Remote run status is $STATUS, not success" >&2; exit 1; }
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${RUNID}_initial_last.pt"
EXPECTED="$(ssh "$M6_HOST" "sha256sum '$REMOTE' | awk '{print \$1}'")"
[[ "$EXPECTED" =~ ^[0-9a-f]{64}$ ]] || { echo "No remote SHA256" >&2; exit 1; }
if [[ -f "$OUT" ]];then
  echo "File exists; verifying (no overwrite): $OUT"
else
  echo "Copying checkpoint (~2.5 GB); preserve host until verification."
  scp "$M6_HOST:$REMOTE" "$OUT.partial"
  mv "$OUT.partial" "$OUT"
fi
ACTUAL="$(sha256sum "$OUT" | awk '{print $1}')"
[[ "$EXPECTED" == "$ACTUAL" ]] || { echo "Checkpoint hash mismatch; do NOT stop before resolving" >&2; exit 1; }
# Stable receipt with important transfer metadata, no checkpoint in metrics ZIP.
"${M6_LOCAL_PYTHON:-python}" - "$OUT_DIR/${RUNID}_checkpoint_receipt.json" "$OUT" "$EXPECTED" "$RUNID" <<'PY'
import json,os,sys,time
p,f,h,runid=sys.argv[1:]
receipt={'run_id':runid,'file':os.path.abspath(f),'sha256':h,
         'bytes':os.path.getsize(f),'verified':True,
         'verified_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
with open(p,'w',encoding='utf-8') as out: json.dump(receipt,out,indent=2)
print('SHA256 VERIFIED: '+f)
print('RECEIPT: '+p)
PY
