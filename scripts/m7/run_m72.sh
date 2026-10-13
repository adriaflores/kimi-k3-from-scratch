#!/usr/bin/env bash
# Remote M7.2 evaluation wrapper. Pinned Python + immutable code, no training steps.
set -Eeuo pipefail
ROOT=/workspace/kimi
ART="$ROOT/artifacts/m7/m72"
PINNED="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
[[ -x "$PINNED" ]] || { echo 'M6 Python missing' >&2;exit 2; }
mkdir -p "$ART" "$ROOT/runs/m7_preparation"
RUNID="m72_pilot_contract_v1_$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$ROOT/runs/m7_preparation/$RUNID"
mkdir "$RUN"
printf '%s\n' "$RUNID" > "$ART/latest_run_id.txt"
printf 'running\n' > "$ART/latest_status.txt"
cp "$ROOT/experiments/m7/current/pilot_80m.yaml" "$RUN/pilot_80m.yaml"
cp "$ROOT/experiments/m7/current/pilot_contract.json" "$RUN/pilot_contract.json"
cp "$ROOT/experiments/m7/current/m6_baseline_train_config.json" "$RUN/m6_baseline_train_config.json"
cp "$ROOT/artifacts/m7/m7_deployed_commit.txt" "$RUN/code_commit.txt" 2>/dev/null || true
finalize() {
  rc="$?"; trap - EXIT
  printf '%s\n' "$rc" > "$RUN/exit_code.txt"
  if python3 "$ROOT/scripts/m7/package_results.py" --run-dir "$RUN" --output "$ART/${RUNID}_results.zip";then
    cp "$ART/${RUNID}_results.zip" "$ART/latest_results.zip"
    sha256sum "$ART/latest_results.zip" | awk '{print $1}' > "$ART/latest_results.sha256"
    if (( rc == 0 ));then printf 'success\n' > "$ART/latest_status.txt";else printf 'failed_%d\n' "$rc" > "$ART/latest_status.txt";fi
  else
    echo 'ERROR: packaging M7.2 failed' >&2
    printf 'package_failed\n' > "$ART/latest_status.txt"
    rc=70
  fi
  exit "$rc"
}
trap finalize EXIT
"$PINNED" "$ROOT/scripts/m7/m72_validate.py" --mode remote --out "$RUN/summary.json" > "$RUN/experiment.log" 2>&1
