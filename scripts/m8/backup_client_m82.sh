#!/usr/bin/env bash
# Run on INDEPENDENT local machine/drive (Windows Git Bash supported).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -n "${M82_LOCAL_BACKUPS:-}" ]] || { echo 'Set M82_LOCAL_BACKUPS to your independent local backup folder' >&2; exit 4; }
exec python scripts/m8/backup_client_m82.py --host "${M8_HOST:-vast}" --directory "$M82_LOCAL_BACKUPS" "$@"
