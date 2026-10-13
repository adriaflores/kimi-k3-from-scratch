#!/usr/bin/env bash
# Local Git Bash. Deploys M7.1-only files and full verified corpus to existing Vast H100.
# Do NOT run until M7.2 preparation has finished and Vast is intentionally started.
set -euo pipefail
M7_HOST="${M7_HOST:-vast}"
[[ "$M7_HOST" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unsafe SSH alias' >&2; exit 1; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DATA_DIR="${M7_DATA_DIR:-C:/datasets/fineweb_edu_6b}"
PYTHON="${M7_LOCAL_PYTHON:-python}"
for path in scripts/m7/upload.sh scripts/m7/launch.sh scripts/m7/status.sh scripts/m7/download.sh scripts/m7/run.sh scripts/m7/m71.py scripts/m7/package_results.py scripts/m7/vendor/branch4_transport.py scripts/m7/vendor/branch4_remote_ops.py experiments/m7/current/run.py experiments/m7/current/experiment.json inputs/m7/reference_metadata.json inputs/m7/origin_inspection/verified_data_manifest.json;do
  [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
if [[ -n "$(git status --porcelain -- scripts/m7 experiments/m7/current inputs/m7)" ]];then
  echo 'M7.1 scripts/experiment/manifest must be committed before paid GPU upload.' >&2
  exit 1
fi
"$PYTHON" - <<'PY'
from pathlib import Path
import json
p=Path('results/m7/m71/local_preflight.json')
if not p.is_file() or json.loads(p.read_text())['status']!='passed':
  raise SystemExit('Run bash scripts/m7/local_preflight.sh successfully first')
PY
if ssh "$M7_HOST" 'tmux has-session -t m7-setup 2>/dev/null';then
  echo 'm7-setup session running: refusing to replace scripts during execution' >&2;exit 1
fi
COMMIT="$(git rev-parse HEAD)"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
printf '%s\n' "$COMMIT" > "$TEMP/m7_deployed_commit.txt"
tar --exclude=__pycache__ --exclude="*.pyc" -czf "$TEMP/m7_bundle.tar.gz" scripts/m7 experiments/m7/current inputs/m7
LOCAL_SHA="$(sha256sum "$TEMP/m7_bundle.tar.gz" | awk '{print $1}')"
ssh "$M7_HOST" 'mkdir -p /workspace/kimi/artifacts/m7 /workspace/kimi/experiments/m7 /workspace/kimi/scripts/m7 /workspace/kimi/ops/m7'
scp "$TEMP/m7_bundle.tar.gz" "$M7_HOST:/workspace/kimi/artifacts/m7/m7_upload.tar.gz"
scp "$TEMP/m7_deployed_commit.txt" "$M7_HOST:/workspace/kimi/artifacts/m7/m7_deployed_commit.txt"
REMOTE_SHA="$(ssh "$M7_HOST" "sha256sum /workspace/kimi/artifacts/m7/m7_upload.tar.gz | awk '{print \$1}'")"
[[ "$LOCAL_SHA" == "$REMOTE_SHA" ]] || { echo 'Remote deployment archive SHA mismatch' >&2; exit 1; }
ssh "$M7_HOST" 'tar -xzf /workspace/kimi/artifacts/m7/m7_upload.tar.gz -C /workspace/kimi && python3 -m py_compile /workspace/kimi/scripts/m7/*.py /workspace/kimi/scripts/m7/vendor/*.py /workspace/kimi/experiments/m7/current/*.py'
echo "Deployed M7.1 from commit $COMMIT. Prechecking GPU/disk."
"$PYTHON" scripts/m7/m71.py remote-precheck --ssh-alias "$M7_HOST"
CONNECTION_PATH="$TEMP/connection.json"
# Git Bash /tmp paths need normalization for native Windows Python.
if command -v cygpath >/dev/null;then CONNECTION_PATH="$(cygpath -w "$TEMP")/connection.json";fi
"$PYTHON" scripts/m7/m71.py make-connection --ssh-alias "$M7_HOST" --output "$CONNECTION_PATH"
"$PYTHON" scripts/m7/m71.py upload --data-dir "$DATA_DIR" --connection "$CONNECTION_PATH"
"$PYTHON" scripts/m7/m71.py remote-postcheck --ssh-alias "$M7_HOST"
echo 'Full dataset transferred/verified; now launch remote M7.1 acceptance via launch.sh.'
