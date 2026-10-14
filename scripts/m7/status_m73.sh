#!/usr/bin/env bash
# Works remotely by ssh; metadata only, never reads 2.5 GB checkpoint.
set -euo pipefail
ROOT=/workspace/kimi
ART="$ROOT/artifacts/m7/m73"
if [[ ! -s "$ART/current_run_id" ]]; then
  echo 'M7.3: no active/past run pointer. Preflight only or not launched.'
  test ! -f "$ART/preflight_only/preflight.json" || cat "$ART/preflight_only/preflight.json"
  exit 0
fi
ID=$(cat "$ART/current_run_id")
OUT="$ART/$ID"
if tmux has-session -t m7-m73 2>/dev/null; then echo 'TMUX: active'; else echo 'TMUX: not active'; fi
echo "RUN_ID: $ID"
echo "STATE: $(cat "$OUT/runner_state.txt" 2>/dev/null || echo unknown)"
echo "STORAGE: $(df -h "$ROOT" | tail -n 1)"
if [[ -f "$ROOT/runs/m73/$ID/metrics.jsonl" ]]; then
  echo 'RECENT TRAINER EVENTS:'
  tail -n 4 "$ROOT/runs/m73/$ID/metrics.jsonl" | cut -c1-600
fi
if [[ -f "$OUT/summary.json" ]]; then
  echo 'SUMMARY:'; cat "$OUT/summary.json"
fi
if [[ -f "$OUT/warmup_receipt.json" ]]; then echo 'WARMUP SAVED: yes'; fi
if [[ -f "$OUT/final_receipt.json" ]]; then echo 'FINAL RECEIPT: available'; fi
if [[ -f "$OUT/m73_results.zip.sha256" ]]; then echo 'RESULTS SHA:'; cat "$OUT/m73_results.zip.sha256"; fi
