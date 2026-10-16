#!/usr/bin/env python3
"""Read-only, dependency-free M9.1 audit of the final Kimi K3 pretraining run.

Writes only under --out. Never imports torch, opens the checkpoint with torch.load,
modifies the training dataset, or contacts GitHub/Vast.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_EVIDENCE_SHA = "10e24124bace12e359f06e11427166cd525e62008906d3ba0ecc0e459ed2f680"
EXPECTED_CHECKPOINT_SHA = "e332bf4d34ea5d392bf0bd6d13a87b90ddc799e737275d972672671492c67dd1"
EXPECTED_CHECKPOINT_BYTES = 2485700551
EXPECTED_FINAL_STEP = 183106
EXPECTED_TOKENS_SEEN = 6000017408
EXPECTED_EVENTS = 3227
EXPECTED_PARAMS = 294676808
EXPECTED_RECIPE_SHA = "2917190c3db675d18f0e439696e77222c87a8e577c5eba96aadaaf4708516ca6"
MANIFEST_RELATIVE = "experiments/m7/m75_gate/evidence/m73_source_manifest.json"
RECIPE_RELATIVE = "experiments/m7/m75_gate/recipe/m8_full_6b.yaml"
CHECKPOINT_WINDOWS = r"C:\datasets\kimi_k3_backups\m8\m8_full_6b_v1\snapshot_step_183106.pt"
BUNDLED_RECIPE = Path(__file__).resolve().parent / "reference" / "m8_full_6b.yaml"
FROZEN_CONTRACT = Path(__file__).resolve().parent / "reference" / "m91_evaluation_contract_frozen.json"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_json(z: zipfile.ZipFile, filename: str) -> dict:
    o = json.loads(z.read(filename))
    if not isinstance(o, dict):
        raise ValueError(f"{filename} is not an object")
    return o


def shortpath(path: Path) -> str:
    return str(path).replace('\\', '/')


def select_evidence(repo: Path, provided: str | None) -> Path:
    if provided:
        return Path(provided).expanduser().resolve()
    return (repo / "results/m8/m83_final/m8_final_evidence.zip").resolve()


def select_checkpoint(provided: str | None) -> Path | None:
    if provided:
        return Path(provided).expanduser()
    candidate = Path(CHECKPOINT_WINDOWS)
    return candidate if candidate.is_file() else None


def verify_source(root: Path, manifest_path: Path | None) -> dict:
    if manifest_path is None or not manifest_path.is_file():
        return {"status": "UNVERIFIED", "reason": "Frozen 35-file M7.3 source manifest unavailable locally", "manifest_expected": MANIFEST_RELATIVE}
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or len(obj) != 35:
        return {"status": "FAIL", "reason": "Manifest should be a dict with exactly 35 files", "entries": len(obj) if isinstance(obj, dict) else None}
    mismatches = []
    for key, value in sorted(obj.items()):
        if not isinstance(key, str) or not isinstance(value, str) or len(value) != 64:
            mismatches.append({"path": str(key), "issue": "invalid_manifest_entry"})
            continue
        relative = Path(key)
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root.resolve()):
            mismatches.append({"path": key, "issue": "unsafe_path"})
        elif not target.is_file():
            mismatches.append({"path": key, "issue": "not_found"})
        else:
            actual = sha256_file(target)
            if actual.lower() != value.lower():
                mismatches.append({"path": key, "issue": "sha256_differs", "expected": value, "actual": actual})
    return {"status": "PASS" if not mismatches else "MISMATCH", "manifest": shortpath(manifest_path), "entries": len(obj), "mismatches": mismatches,
            "warning": "PASS compares local repo files with frozen manifest; it does not by itself prove the exact runtime source was identical."}


def weighted_summary(evidence: Path, out: Path) -> tuple[dict, list[dict]]:
    all_events = Counter()
    val_events: list[dict] = []
    updates = clipped = skipped = nonfinite = qk_clipped = qk_observed = 0
    weighted_loss_sum = weighted_grad_sum = weighted_train_seconds = 0.0
    train_tokens = 0
    qk_max_logit = -math.inf
    final_step = final_tokens = None
    first_step = first_tokens = None
    first_session_start = None
    training_last_step = None
    ordered_steps_ok = True
    train_rows = 0
    cumulative_ends: list[tuple[int, int]] = []
    max_seen_session = None
    with zipfile.ZipFile(evidence) as z, (out / "train_windows.csv").open("w", newline="", encoding="utf-8") as train_f, (out / "validation.csv").open("w", newline="", encoding="utf-8") as val_f:
        tr = csv.writer(train_f); vl = csv.writer(val_f)
        tr.writerow(["step", "tokens_seen", "updates", "tokens", "train_loss", "lr", "grad_norm_mean", "grad_norm_max", "clipped_fraction", "qk_max_logit", "qk_clipped_head_updates", "train_tokens_per_second"])
        vl.writerow(["step", "tokens_seen", "val_loss", "eval_seconds"])
        with z.open("metrics.jsonl") as f:
            for line_no, raw in enumerate(f, 1):
                evt = json.loads(raw)
                name = evt["event"]
                all_events[name] += 1
                final_step, final_tokens = evt["step"], evt["tokens_seen"]
                max_seen_session = evt.get("session_id")
                if first_step is None:
                    first_step, first_tokens = final_step, final_tokens
                if name == "session_start":
                    if first_session_start is None:
                        first_session_start = evt.get("timestamp")
                if name == "train":
                    u = int(evt["updates"])
                    a = int(evt["attempted_updates"])
                    assert 0 < u <= a, f"Invalid update count at line {line_no}"
                    clipped_window = float(evt["clipped_fraction"]) * a
                    if not math.isclose(clipped_window, round(clipped_window), abs_tol=1e-5):
                        raise ValueError(f"Fractional clip count at line {line_no}")
                    updates += u
                    clipped += round(clipped_window)
                    skipped += int(evt["skipped_updates_in_window"])
                    nonfinite += int(evt["nonfinite_grad_norms"])
                    train_tokens += int(evt["tokens"])
                    weighted_loss_sum += float(evt["train_loss"]) * u
                    weighted_grad_sum += float(evt["grad_norm_mean"]) * a
                    weighted_train_seconds += float(evt["train_seconds"])
                    qk = evt.get("qk_clip", {})
                    qk_clipped += int(qk.get("clipped_head_updates", 0))
                    qk_observed += int(qk.get("observed_head_updates", 0))
                    qk_max_logit = max(qk_max_logit, float(qk.get("max_logit", -math.inf)))
                    if training_last_step is not None and evt["step"] <= training_last_step:
                        ordered_steps_ok = False
                    training_last_step = evt["step"]
                    cumulative_ends.append((evt["step"], u))
                    train_rows += 1
                    tr.writerow([evt["step"], evt["tokens_seen"], u, evt["tokens"], evt["train_loss"], evt["lr"], evt["grad_norm_mean"], evt["grad_norm_max"], evt["clipped_fraction"], qk.get("max_logit"), qk.get("clipped_head_updates"), evt["train_tokens_per_second"]])
                elif name == "val":
                    val_events.append({"step": int(evt["step"]), "tokens_seen": int(evt["tokens_seen"]), "loss": float(evt["val_loss"])})
                    vl.writerow([evt["step"], evt["tokens_seen"], evt["val_loss"], evt["eval_seconds"]])
    # Last ~8192 updates, respecting partial windows: recover integer clipped counts for intervals entirely included.
    # Avoid interpreting clipping fraction of a two-update final window as a full-run statistic.
    tail_events = []
    with zipfile.ZipFile(evidence) as z:
        for line in z.open("metrics.jsonl"):
            e = json.loads(line)
            if e["event"] == "train" and e["step"] > EXPECTED_FINAL_STEP - 8192:
                tail_events.append((int(e["attempted_updates"]), round(float(e["clipped_fraction"]) * int(e["attempted_updates"]))))
    tail_u = sum(u for u, c in tail_events); tail_c = sum(c for u, c in tail_events)
    best = min(val_events, key=lambda v: v["loss"]) if val_events else None
    last = val_events[-1] if val_events else None
    result = {
        "event_counts": dict(all_events), "events_total": sum(all_events.values()),
        "first_step": first_step, "first_tokens_seen": first_tokens, "last_step": final_step, "last_tokens_seen": final_tokens,
        "updates_logged_in_m8": updates, "updates_expected_m8": EXPECTED_FINAL_STEP - first_step,
        "ordered_training_steps": ordered_steps_ok, "training_windows": train_rows,
        "training_tokens_logged_in_m8": train_tokens, "clipped_updates": clipped,
        "clipped_fraction_m8": clipped / updates if updates else None,
        "tail_updates": tail_u, "tail_clipped_updates": tail_c, "tail_clipped_fraction": tail_c / tail_u if tail_u else None,
        "skipped_updates_in_windows": skipped, "nonfinite_grad_norms": nonfinite,
        "weighted_train_loss_m8_not_a_benchmark": weighted_loss_sum / updates if updates else None,
        "weighted_mean_grad_norm": weighted_grad_sum / updates if updates else None,
        "train_compute_seconds_logged_m8": weighted_train_seconds,
        "train_throughput_tokens_per_train_second_m8": train_tokens / weighted_train_seconds if weighted_train_seconds else None,
        "qk_clipped_head_updates": qk_clipped, "qk_observed_head_updates": qk_observed,
        "qk_max_logit_over_logged_windows": qk_max_logit,
        "validation_measurements": len(val_events), "best_validation": best, "final_validation": last,
        "last_session_id": max_seen_session, "first_session_start_timestamp": first_session_start,
    }
    return result, val_events


def frozen_contract(run: dict, final_audit: dict, evidence_sha: str) -> dict:
    session = run["sessions"][-1]
    val_tokens = int(session["data"]["val"]["total_tokens"])
    length = int(session["train_config"]["seq_len"])
    # 64 nonoverlapping deterministic windows, scattered across held-out validation beyond monitoring prefix.
    n, first, stride = 64, 1_000_000, 4_800_000
    windows = [{"offset": first + i * stride, "input_tokens": length, "target_tokens": length} for i in range(n)]
    if windows[-1]["offset"] + length >= val_tokens:
        raise ValueError("Pre-registered windows exceed validation corpus")
    return {
        "schema_version": "m91_eval_contract_1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "baseline_checkpoint_sha256": final_audit["checkpoint_sha256"],
        "evidence_zip_sha256": evidence_sha,
        "model_config": session["model_config"],
        "model_config_json_sha256": canonical_sha(session["model_config"]),
        "data": session["data"],
        "tokenizer": session["data"]["val"]["tokenizer"],
        "seed": 1337,
        "evaluation": {
            "kind": "causal-language-model-contiguous-stream-NLL-per-token",
            "split": "val", "data_total_tokens": val_tokens,
            "source": "tokenized val stream prepared in M7.1; read only; no use of repeated first-262144-token monitoring prefix",
            "input_alignment": "x = tokens[offset : offset+1024]; y = tokens[offset+1 : offset+1025]; every target counted exactly once",
            "n_windows": n, "window_length": length,
            "total_scored_tokens": n * length,
            "window_offsets": [w["offset"] for w in windows],
            "reset_state_each_window": True,
            "aggregation": "sum of per-token negative log likelihood / scored token count; perplexity=exp(mean NLL)",
            "report": ["mean_nll_nats_per_token", "perplexity_same_tokenizer", "number_scored_tokens", "per_window_nll", "runtime", "weights_sha256", "code_provenance"],
            "uncertainty": "optional bootstrap resampling of the 64 pre-registered window losses; label it approximate and dependent on corpus structure",
            "compare_models": "only compare per-token PPL using same tokenizer and same token stream; otherwise avoid direct PPL comparisons",
        },
        "qualitative_protocol": {
            "kind": "base-LM continuation, not chat instruction following",
            "selection": "fixed prompts and seeds committed before inspection of outputs",
            "generation": "first full-prefix greedy without cache; add seeded top-k/top-p if smoke passes",
            "limits": {"preferred_context_length": length, "max_new_tokens": 64},
        },
        "decision_rule": "GO continued pretraining only if model passes inference/held-out checks and expected learning benefit justifies fresh data or another epoch; otherwise HOLD/STOP with documented results",
    }


def run_audit(args: argparse.Namespace) -> dict:
    repo = Path(args.repo).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else repo / "results" / "m9" / "m91_audit"
    if not repo.is_dir():
        raise FileNotFoundError(f"Repo missing: {repo}")
    out.mkdir(parents=True, exist_ok=True)
    evidence = select_evidence(repo, args.evidence)
    if not evidence.is_file():
        raise FileNotFoundError(f"Evidence ZIP missing: {evidence}; supply --evidence ...")
    evidence_sha = sha256_file(evidence)
    checks: dict[str, str] = {}
    checks["evidence_zip_sha256"] = "PASS" if evidence_sha == EXPECTED_EVIDENCE_SHA else "FAIL"
    with zipfile.ZipFile(evidence) as z:
        required = {"final_audit.json", "run.json", "metrics.jsonl", "final_snapshot_receipt.json", "offsite_ack.json", "runner_state.json", "probe_receipt.json", "recovery_gate.json"}
        missing = sorted(required - set(z.namelist()))
        checks["required_evidence_files"] = "PASS" if not missing else "FAIL"
        if missing:
            raise ValueError(f"Evidence ZIP lacks required files: {missing}")
        final = load_json(z, "final_audit.json")
        receipt = load_json(z, "final_snapshot_receipt.json")
        ack = load_json(z, "offsite_ack.json")
        runner = load_json(z, "runner_state.json")
        run = load_json(z, "run.json")
    checks["final_audit"] = "PASS" if final.get("status") == "PASS" and final.get("final_step") == EXPECTED_FINAL_STEP and final.get("tokens_seen") == EXPECTED_TOKENS_SEEN and final.get("events") == EXPECTED_EVENTS else "FAIL"
    checks["checkpoint_receipt_consistency"] = "PASS" if all([
        final.get("checkpoint_sha256") == receipt.get("sha256") == ack.get("sha256") == EXPECTED_CHECKPOINT_SHA,
        final.get("checkpoint_bytes") == receipt.get("bytes") == ack.get("bytes") == EXPECTED_CHECKPOINT_BYTES,
        receipt.get("step") == ack.get("step") == EXPECTED_FINAL_STEP]) else "FAIL"
    checks["trainer_final_state"] = "PASS" if runner.get("train_exit_code") == 0 and runner.get("incident") is None and runner.get("warnings") == [] else "FAIL"
    same_model = bool(run.get("sessions")) and all(s.get("model_config") == run["sessions"][-1].get("model_config") for s in run["sessions"])
    same_training = bool(run.get("sessions")) and all(s.get("train_config") == run["sessions"][-1].get("train_config") for s in run["sessions"])
    checks["consistent_session_config"] = "PASS" if same_model and same_training else "FAIL"
    checks["parameter_count"] = "PASS" if all(s.get("parameter_count") == EXPECTED_PARAMS for s in run["sessions"]) else "FAIL"
    metric, vals = weighted_summary(evidence, out)
    checks["metrics_complete"] = "PASS" if all([
        metric["events_total"] == EXPECTED_EVENTS,
        metric["last_step"] == EXPECTED_FINAL_STEP,
        metric["last_tokens_seen"] == EXPECTED_TOKENS_SEEN,
        metric["updates_logged_in_m8"] == metric["updates_expected_m8"],
        metric["ordered_training_steps"],
        metric["skipped_updates_in_windows"] == 0,
        metric["nonfinite_grad_norms"] == 0,
        metric["final_validation"] is not None and metric["final_validation"]["step"] == EXPECTED_FINAL_STEP,
    ]) else "FAIL"
    ckpt = select_checkpoint(args.checkpoint)
    ckpt_info = {"status": "UNVERIFIED", "reason": "Checkpoint unavailable on this machine; pass --checkpoint Windows path"}
    if ckpt is not None and ckpt.is_file():
        size = ckpt.stat().st_size
        actual = sha256_file(ckpt) if size == EXPECTED_CHECKPOINT_BYTES else None
        ckpt_info = {"status": "PASS" if actual == EXPECTED_CHECKPOINT_SHA else "FAIL", "size_bytes": size, "sha256": actual, "expected_size_bytes": EXPECTED_CHECKPOINT_BYTES, "expected_sha256": EXPECTED_CHECKPOINT_SHA}
    checks["checkpoint_local_sha256"] = ckpt_info["status"]
    manifest_path = Path(args.manifest).expanduser() if args.manifest else repo / MANIFEST_RELATIVE
    source_root = Path(args.reference).expanduser().resolve() if args.reference else repo
    source_info = verify_source(source_root, manifest_path)
    source_info["source_root"] = shortpath(source_root)
    checks["frozen_source_manifest"] = source_info["status"]
    recipe = repo / RECIPE_RELATIVE
    if not recipe.is_file() and BUNDLED_RECIPE.is_file():
        recipe = BUNDLED_RECIPE
    recipe_info = {"status": "UNVERIFIED", "reason": "Approved recipe unavailable at repo path and in bundled reference"}
    if recipe.is_file():
        actual = sha256_file(recipe)
        recipe_info = {"status": "PASS" if actual == EXPECTED_RECIPE_SHA else "MISMATCH", "path": shortpath(recipe), "sha256": actual, "expected_sha256": EXPECTED_RECIPE_SHA}
    checks["approved_recipe"] = recipe_info["status"]
    # A local checkout that differs from the frozen production snapshot does not
    # invalidate the run; it blocks inference until exact source is recovered.
    contract = frozen_contract(run, final, evidence_sha)
    if FROZEN_CONTRACT.is_file():
        checks["frozen_evaluation_contract"] = "PASS" if json.loads(FROZEN_CONTRACT.read_text(encoding="utf-8")) == contract else "FAIL"
    else:
        checks["frozen_evaluation_contract"] = "FAIL"
    # Final status after immutable contract check.
    hard_fail = any(v in {"FAIL", "MISMATCH"} for k, v in checks.items() if k != "frozen_source_manifest")
    remaining = [k for k, v in checks.items() if v in {"UNVERIFIED", "MISMATCH"}]
    status = "FAIL" if hard_fail else "PASS" if not remaining else "HOLD_PROVENANCE"
    (out / "evaluation_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "schema_version": "m91_audit_1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "checks": checks, "unverified_items": remaining,
        "evidence_sha256": evidence_sha, "checkpoint": ckpt_info, "source_manifest": source_info,
        "approved_recipe": recipe_info, "final_audit": final, "train_metrics": metric,
        "run_id_internal": run.get("run_id"), "run_sessions": len(run["sessions"]),
        "reference_config": run["sessions"][-1]["model_config"],
        "train_recipe": run["sessions"][-1]["train_config"],
        "tokenizer": run["sessions"][-1]["data"]["val"]["tokenizer"],
        "val_total_tokens": run["sessions"][-1]["data"]["val"]["total_tokens"],
        "notes": ["M8 metrics cover steps 2443..183106; the first 2442 updates were performed in the M7 pilot.",
                  "The monitoring val loss is NOT an independent held-out M9 benchmark.",
                  "HOLD_PROVENANCE means that checks could not be completed on this machine, not that the run is invalid.",
                  "No checkpoint contents were deserialized and no training/inference was performed."],
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    loss = metric["final_validation"]["loss"] if metric["final_validation"] else float("nan")
    summary = [
        "# M9.1 — Auditoría de procedencia y contrato", "",
        f"**Estado: {status}**", "",
        "## Verificaciones", "",
        "| Control | Estado |", "|---|---|", *[f"| `{key}` | {value} |" for key, value in checks.items()], "",
        "## Métricas M8 (no son resultados M9)", "",
        f"- Eventos: {metric['events_total']} ({metric['training_windows']} train; {metric['validation_measurements']} val).",
        f"- Updates observados en M8: {metric['updates_logged_in_m8']} (a partir del step {metric['first_step']}).",
        f"- Gradient clipping: {metric['clipped_updates']} / {metric['updates_logged_in_m8']} = {metric['clipped_fraction_m8']:.4%}.",
        f"- Últimos {metric['tail_updates']} updates registrados: {metric['tail_clipped_fraction']:.4%} clipping.",
        f"- QK clipped head updates: {metric['qk_clipped_head_updates']} / {metric['qk_observed_head_updates']}.",
        f"- Best monitoring validation loss: {metric['best_validation']}",
        f"- Final monitoring validation loss: {loss:.6f} (prefijo recurrente, no benchmark independiente).", "",
        "## Protocolo pre-registrado", "",
        f"- 64 ventanas separadas de {contract['evaluation']['window_length']} tokens de entrada; {contract['evaluation']['total_scored_tokens']} tokens puntuados.",
        "- Offsets sobre el stream val: 1.000.000 + 4.800.000 × i, i=0..63; excluyen el prefijo repetido de monitorización.",
        "- Targets desplazados 1 token, sin padding ni estado entre ventanas.",
        "- NLL/token + perplexity del MISMO tokenizador; pérdida por ventana para incertidumbre.", "",
        "## Pendientes", "",
        *( [f"- `{item}`: evidencia no comprobada en este equipo." for item in remaining] if remaining else ["- Ninguna comprobación de procedencia pendiente."]),
        "", "**Nota:** la verificación del manifiesto M7.3 exige el fichero original de 35 hashes; nunca crear uno a partir del código actual y llamarlo baseline.", "",
    ]
    (out / "M91_READOUT.md").write_text("\n".join(summary), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Local checkout root of kimi-k3-from-scratch")
    parser.add_argument("--evidence", default=None, help="Path to original m8_final_evidence.zip")
    parser.add_argument("--checkpoint", default=None, help="Read-only SHA-256 of local 2.49GB .pt file (not deserialized)")
    parser.add_argument("--manifest", default=None, help="Optional original M7.3 35-file frozen source manifest")
    parser.add_argument("--reference", default=None, help="Optional directory of exact source snapshot from H100; default compares current repo")
    parser.add_argument("--out", default=None, help="Output folder (default results/m9/m91_audit)")
    args = parser.parse_args(argv)
    try:
        result = run_audit(args)
    except (OSError, ValueError, KeyError, AssertionError, zipfile.BadZipFile) as exc:
        print(f"M9.1 AUDIT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"M9.1 STATUS: {result['status']}")
    for key, val in result["checks"].items():
        print(f"  {key}: {val}")
    output_dir = Path(args.out).expanduser().resolve() if args.out else Path(args.repo).expanduser().resolve() / "results/m9/m91_audit"
    print(f"REPORT: {output_dir / 'M91_READOUT.md'}")
    print(f"JSON: {output_dir / 'audit.json'}")
    return 2 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
