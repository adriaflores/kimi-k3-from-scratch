---
title: "RunContext and start_run"
description: "Records reproducible run provenance, session identities and resume-compatible run manifests."
sourcePath: "src/kimi_k3/training/run.py"
status: published
---

`start_run(...)` creates or extends a **run manifest**, rather than training a model. It receives an already constructed model, optimizer, effective training configuration and loader state, and returns `RunContext(directory,checkpoint_path,run_id,session_id)` [source L25–L30](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L25-L30). An individual run may contain multiple sessions; every resume gets a new session ID while retaining the run ID. The checkpoint's numerical restoration belongs to `trainer.py`, and CLI argument parsing belongs to the training script.

## Capture Git and environment provenance

`_git_metadata(repo_root)` [source L33–L48](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L33-L48) invokes `git rev-parse HEAD` and `git status --porcelain` with bounded subprocess timeouts. On unavailable Git it records `commit=None` and `dirty=None`, rather than inventing a revision. The environment record [source L51–L74](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L51-L74) includes Python/platform, NumPy and PyTorch versions, CUDA/cuDNN availability, CPU count, device and GPU properties when applicable. It adds runtime-signature fields from `trainer.py` plus matrix precision and TF32 flags. These are **descriptive metadata**, not guarantees of deterministic execution.

`_dataset_metadata(loader)` [source L77–L85](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L77-L85) records resolved prepared-data directory, manifest digest, split token count, source and tokenizer metadata. It does not rehash every shard's contents: the metadata digest identifies the normalized manifest, not an independent corpus audit. `_optimizer_metadata` [source L107–L116](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L107-L116) records the optimizer class, per-parameter layout, group options and parameter/element counts without embedding tensor values in JSON.

## Choose a run directory and prevent accidental overwrite

At entry, `start_run` [source L119–L149](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L119-L149) normalizes `resume_path`. If `run_dir` is omitted but `run.json` is beside the checkpoint, it reuses that directory. Otherwise a fresh invocation chooses `results/training/runs/<UTC timestamp>-<runid prefix>`. When the chosen directory already exists, `_load_existing_manifest` [source L88–L104](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L88-L104) demands a supported manifest and a checkpoint associated with one of its previous sessions. Merely pointing at an unrelated checkpoint is not permission to overwrite an existing run.

The checkpoint defaults to an existing run's resume checkpoint, or a new run's `last.pt`; an explicit path is also supported. The manifest and `metrics.jsonl` paths are protected against checkpoint-name collisions. Paths inside the run directory are additionally recorded relatively, so a self-contained run folder may be moved without losing the association.

## Session payload and resume invariants

Each new session [source L150–L173](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L150-L173) stores a UUID, start time, Git provenance, model/config, parameter count, effective train recipe, QK-Clip policy and head geometry, optimizer grouping, environment, train/val manifests, initial step/tokens/skips and loader cursor states. Its RNG source is either `seed` or `checkpoint`, and it names the resume and output checkpoints. Tuples/dtypes are normalized to plain JSON values before recording.

If extending an existing run, the code compares the **model configuration and both dataset metadata digests** with the preceding session [source L174–L184](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L174-L184); a difference aborts the append. It does not claim all runtime/optimizer conditions match merely because the manifest check passed: tighter compatibility and RNG restoration happen in `load_checkpoint`. After these validations, the new manifest is written to a session-specific temporary file and atomically replaced [source L185–L192](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py#L185-L192). A fresh directory is created only for a new run.

## Verification and boundary

[test test_run.py L41–L87](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_run.py#L41-L87) checks the manifest fields and that provenance capture does not mutate the live model, loaders or RNG. [test test_run.py L90–L118](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_run.py#L90-L118) checks resume run-ID continuity and session-ID uniqueness; [test test_run.py L120–L171](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_run.py#L120-L171) checks existing-run safety and relocation. [test test_checkpoint_resume.py L537–L628](https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py#L537-L628) exercises manifest/session behavior during actual resumed training. Assertions were inspected, not rerun. A `run.json` by itself cannot establish successful model convergence.

<aside class="reading-sources" aria-label="Sources and verification">
  <span class="reading-note-label">Sources &amp; verification</span>
  <dl>
    <dt>Source</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/src/kimi_k3/training/run.py">run.py · pinned source revision</a></dd>
    <dt>Tests</dt><dd><a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/unit/training/test_run.py">test_run.py</a>; <a href="https://github.com/adriaflores/kimi-k3-from-scratch/blob/10c0dcfa8d2b0415f8866674effd82b7548772e7/tests/integration/test_checkpoint_resume.py">test_checkpoint_resume.py</a>. Assertions inspected; no fresh execution claimed.</dd>
  </dl>
</aside>
