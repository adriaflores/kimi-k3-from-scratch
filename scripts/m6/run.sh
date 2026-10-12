#!/usr/bin/env bash
# Remote. Generic experiment runner and result collector. Do not launch directly on PC.
set -Eeuo pipefail
ROOT=/workspace/kimi
ART="$ROOT/artifacts"
PYTHON="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
ENTRY="$ROOT/experiments/m6/current/run.py"
META="$ROOT/experiments/m6/current/experiment.json"
mkdir -p "$ART/m6" "$ROOT/runs/m6_experiments"
[[ -x "$PYTHON" && -f "$ENTRY" && -f "$META" ]] || { echo 'Missing pinned runtime or experiment. Recheck paths.' >&2; exit 3; }
EXPID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_id"])' "$META")"
[[ "$EXPID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe experiment_id' >&2; exit 4; }
RUNID="${EXPID}_$(date -u +%Y%m%dT%H%M%SZ)"
RUNDIR="$ROOT/runs/m6_experiments/$RUNID"
mkdir "$RUNDIR"
cp "$META" "$RUNDIR/experiment.json"
cp "$ART/m6_deployed_commit.txt" "$RUNDIR/code_commit.txt" 2>/dev/null || true
hostname > "$RUNDIR/hostname.txt"
(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true) > "$RUNDIR/gpu.txt" 2>&1
printf '%s\n' "$RUNID" > "$ART/m6_latest_run_id.txt"
printf 'running\n' > "$ART/m6_latest_status.txt"
finalize() {
  rc="$?"; trap - EXIT
  printf '%s\n' "$rc" > "$RUNDIR/exit_code.txt"
  (nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true) > "$RUNDIR/gpu_end.txt" 2>&1
  if python3 "$ROOT/scripts/m6/package_results.py" --run-dir "$RUNDIR" --output "$ART/m6/${RUNID}_results.zip"; then
    cp "$ART/m6/${RUNID}_results.zip" "$ART/m6_latest_results.zip"
    sha256sum "$ART/m6_latest_results.zip" | awk '{print $1}' > "$ART/m6_latest_results.sha256"
    if [[ "$rc" -eq 0 ]]; then printf 'success\n' > "$ART/m6_latest_status.txt"; else printf 'failed_%d\n' "$rc" > "$ART/m6_latest_status.txt"; fi
    echo "RESULT ZIP: $ART/m6/${RUNID}_results.zip"
  else
    echo 'ERROR packaging results'; printf 'package_failed\n' > "$ART/m6_latest_status.txt"; rc=70
  fi
  exit "$rc"
}
trap finalize EXIT
printf 'Running %s on %s\n' "$RUNID" "$(date -u)"
"$PYTHON" "$ENTRY" --out "$RUNDIR/data" > "$RUNDIR/experiment.log" 2>&1
