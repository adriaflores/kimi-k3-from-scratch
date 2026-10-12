#!/usr/bin/env bash
# Run in Git Bash at any path within the local Git repository.
set -euo pipefail
M6_HOST="${M6_HOST:-vast-kimi}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
for path in scripts/m6/upload.sh scripts/m6/run.sh scripts/m6/launch.sh scripts/m6/status.sh scripts/m6/download.sh scripts/m6/package_results.py experiments/m6/current/run.py experiments/m6/current/experiment.json; do
  test -f "$path" || { echo "Missing $path" >&2; exit 1; }
done
if test -n "$(git status --porcelain -- scripts/m6 experiments/m6/current)"; then
  echo 'Experiment/scripts not committed. Commit these files first to ensure a reproducible revision.' >&2
  exit 1
fi
COMMIT="$(git rev-parse HEAD)"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
printf '%s\n' "$COMMIT" > "$TEMP/m6_commit.txt"
tar -czf "$TEMP/m6_bundle.tar.gz" scripts/m6 experiments/m6/current
sha256sum "$TEMP/m6_bundle.tar.gz"
ssh "$M6_HOST" 'mkdir -p /workspace/kimi/artifacts /workspace/kimi/experiments/m6 /workspace/kimi/scripts/m6'
scp "$TEMP/m6_bundle.tar.gz" "$M6_HOST:/workspace/kimi/artifacts/m6_upload.tar.gz"
scp "$TEMP/m6_commit.txt" "$M6_HOST:/workspace/kimi/artifacts/m6_deployed_commit.txt"
ssh "$M6_HOST" 'tar -xzf /workspace/kimi/artifacts/m6_upload.tar.gz -C /workspace/kimi && python3 -m py_compile /workspace/kimi/experiments/m6/current/*.py /workspace/kimi/scripts/m6/package_results.py'
echo "Uploaded experiment from commit $COMMIT to $M6_HOST"
