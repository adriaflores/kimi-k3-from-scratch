#!/usr/bin/env bash
# Internal tmux worker. NEVER use training CLI or run M8.
set -uo pipefail
BASE=/workspace/kimi
RUN="${1:?expected run folder}"
RUN_ID="$(basename "$RUN")"
KIT="$BASE/experiments/m7/m75_gate"
PYTHON="$BASE/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
ARTIFACT_DIR="$BASE/artifacts/m7/m75/$RUN_ID"
mkdir -p "$RUN" "$ARTIFACT_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
printf '{"run_id":"%s","purpose":"M7.5 real load_checkpoint gate; no optimizer steps","started_at_utc":"%s","source_checkpoint":"%s"}\n' \
  "$RUN_ID" "$(date -u +%FT%TZ)" "$BASE/runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt" > "$RUN/run_info.json"
{
  date -u +'%FT%TZ'
  nvidia-smi -L
  df -h "$BASE"
  "$PYTHON" -V
  "$PYTHON" "$KIT/scripts/probe_resume_m75.py" --receipt "$RUN/m75_resume_receipt.json"
} > "$RUN/m75_probe.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
# Failing pre-Python checks also leave a retrievable failure record, not a missing archive.
if [[ ! -f "$RUN/m75_resume_receipt.json" ]]; then
  printf '{"status":"FAIL","error":"Remote runner exited before receipt was produced","exit_code":%d}\n' "$RC" > "$RUN/m75_resume_receipt.json"
fi
export M75_RUN_DIR="$RUN" M75_ARTIFACT_DIR="$ARTIFACT_DIR" M75_RUN_ID="$RUN_ID"
# No checkpoint, corpus or credentials in results. Python ZIP avoids requiring 'zip' binary.
"${PYTHON}" - <<'PY'
import hashlib,os,zipfile
from pathlib import Path
run=Path(os.environ['M75_RUN_DIR'])
art=Path(os.environ['M75_ARTIFACT_DIR'])
name=os.environ['M75_RUN_ID']+'_results.zip'
out=art/name
files=('run_info.json','m75_probe.log','m75_resume_receipt.json','exit_code.txt')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for item in files:
        z.write(run/item,arcname=item)
h=hashlib.sha256(out.read_bytes()).hexdigest()
(art/(name+'.sha256')).write_text(f'{h}  {name}\n',encoding='utf-8')
print(f'M7.5 artifact: {out}; sha256={h}')
PY
ZIP_RC=$?
if [[ "$ZIP_RC" != 0 ]]; then
  echo "WARNING: failed to package result ZIP (exit $ZIP_RC)" >> "$RUN/m75_probe.log"
fi
if [[ "$RC" == 0 && "$ZIP_RC" == 0 ]]; then
  echo 'M7.5 PASS: load_checkpoint verified; NO training updates.'
else
  echo "M7.5 FAIL: probe exit $RC; artifact packaging exit $ZIP_RC."
fi
if [[ "$RC" != 0 ]]; then exit "$RC"; fi
exit "$ZIP_RC"
