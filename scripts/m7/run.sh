#!/usr/bin/env bash
# Remote: run only data/runtime acceptance; no train steps and no checkpoint changes.
set -Eeuo pipefail
ROOT=/workspace/kimi
ART="$ROOT/artifacts/m7"
ENTRY="$ROOT/experiments/m7/current/run.py"
META="$ROOT/experiments/m7/current/experiment.json"
mkdir -p "$ART" "$ROOT/runs/m7_preparation"
[[ -f "$ENTRY" && -f "$META" ]] || { echo 'Missing M7.1 experiment' >&2;exit 3; }
EXPID="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["experiment_id"])' "$META")"
[[ "$EXPID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe experiment_id' >&2;exit 4; }
RUNID="${EXPID}_$(date -u +%Y%m%dT%H%M%SZ)"
RUNDIR="$ROOT/runs/m7_preparation/$RUNID"
mkdir "$RUNDIR"
cp "$META" "$RUNDIR/experiment.json"
cp "$ART/m7_deployed_commit.txt" "$RUNDIR/code_commit.txt" 2>/dev/null || true
hostname > "$RUNDIR/hostname.txt"
(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true) > "$RUNDIR/gpu.txt" 2>&1
printf '%s\n' "$RUNID" > "$ART/latest_run_id.txt"
printf 'running\n' > "$ART/latest_status.txt"
finalize() {
  rc="$?";trap - EXIT
  printf '%s\n' "$rc" > "$RUNDIR/exit_code.txt"
  if python3 "$ROOT/scripts/m7/package_results.py" --run-dir "$RUNDIR" --output "$ART/${RUNID}_results.zip";then
    cp "$ART/${RUNID}_results.zip" "$ART/latest_results.zip"
    sha256sum "$ART/latest_results.zip" | awk '{print $1}' > "$ART/latest_results.sha256"
    if [[ "$rc" -eq 0 ]];then printf 'success\n' > "$ART/latest_status.txt";else printf 'failed_%d\n' "$rc" > "$ART/latest_status.txt";fi
    echo "RESULT ZIP: $ART/${RUNID}_results.zip"
  else
    echo 'ERROR packaging results' >&2;printf 'package_failed\n' > "$ART/latest_status.txt";rc=70
  fi
  exit "$rc"
}
trap finalize EXIT
python3 "$ENTRY" --out "$RUNDIR/data" > "$RUNDIR/experiment.log" 2>&1
