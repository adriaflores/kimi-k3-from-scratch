#!/usr/bin/env bash
# Remote gated tmux launcher: DRILL stops after first checkpoint; CONTINUOUS only after M8.3.
set -euo pipefail
ROOT=/workspace/kimi
PY="$ROOT/control/kimi_k3_milestone5_branch5/environments/milestone5_linux/.venv/bin/python"
PHASE="${1:-}"
[[ "$PHASE" == drill || "$PHASE" == continuous ]] || exit 4
[[ "${M82_START:-}" == "YES_M8_${PHASE^^}" ]] || { echo 'NO-GO: explicit phase authorization missing' >&2; exit 4; }
test -x "$PY" || { echo 'NO-GO: pinned Python absent' >&2; exit 4; }
command -v tmux >/dev/null || { echo 'NO-GO: tmux absent' >&2; exit 4; }
# Gate before creating new run/artifact paths.
"$PY" "$ROOT/scripts/m8/m82.py" guard --phase "$PHASE"
if [[ "$PHASE" == drill ]]; then
  SOURCE="$ROOT/runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt"
  EXPECTED=cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12
  ACTUAL="$(sha256sum "$SOURCE" | awk '{print $1}')"
  [[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'NO-GO: original M7.3 checkpoint altered since preflight' >&2; exit 4; }
fi
if tmux has-session -t kimi_m82 2>/dev/null; then echo 'NO-GO: M8 already running' >&2; exit 4; fi
# Guard existing finished states to avoid accidentally rerunning continuous.
if [[ "$PHASE" == continuous ]]; then
  STATE="$ROOT/artifacts/m8/m82/m8_full_6b_v1/runner_state.json"
  "$PY" - "$STATE" <<'PY'
import json,sys
from pathlib import Path
state=json.loads(Path(sys.argv[1]).read_text())['state']
if state not in ('AWAITING_M83_RECOVERY_GATE', 'RECOVERY_APPROVED'):
    raise SystemExit('NO-GO: state is neither drill finished nor incident recovery approved')
PY
fi
mkdir -p "$ROOT/artifacts/m8/m82/m8_full_6b_v1"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
# Only remote-controlled training path, detached from SSH, no login shell dependencies.
tmux new-session -d -s kimi_m82 "'$PY' -u '$ROOT/scripts/m8/m82.py' supervise --phase '$PHASE' --python '$PY'"
echo "M8.2 LAUNCHED: $PHASE. Review status; never delete Vast before independent backup."
