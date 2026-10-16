"""M9.2 diagnostic for causal suffix changes. Read-only, uses model.eval().

This module measures, never edits, model parameters. A causal network may
show small floating-point differences when MoE grouping sizes change. A large
or systematic change must not be waived by relaxing a tolerance.
"""
from __future__ import annotations

import torch


def capture_prefix(model, ids, *, device, precision, suffix_token, inference_context, chunk_size=64):
    """Capture outputs from the embedding, attention, FFN, and each layer.

    Only a *small* prefix is retained (CPU float32) to bound GPU allocations.
    Test sequences have the same length and differ only after the prefix.
    """
    prefix_len = len(ids)
    if not (1 <= prefix_len <= chunk_size):
        raise ValueError('Diagnostic prefix must fit within a KDA chunk')
    length = ((prefix_len + chunk_size - 1) // chunk_size) * chunk_size
    if suffix_token < 0 or suffix_token >= model.vocab_size:
        raise ValueError('Invalid diagnostic suffix token')
    tokens = torch.tensor([list(ids) + [suffix_token] * (length - prefix_len)], device=device)
    snapshots = {}
    handles = []

    def mk_hook(name):
        def hook(_module, _inputs, out):
            tensor = out[0] if isinstance(out, tuple) else out
            snapshots[name] = tensor[:, :prefix_len].detach().float().cpu().clone()
        return hook

    handles.append(model.embed_tokens.register_forward_hook(mk_hook('embed_tokens')))
    for i, layer in enumerate(model.layers):
        handles.append(layer.attn.register_forward_hook(mk_hook(f'layer_{i:02d}_attention')))
        handles.append(layer.ffn.register_forward_hook(mk_hook(f'layer_{i:02d}_ffn')))
        handles.append(layer.register_forward_hook(mk_hook(f'layer_{i:02d}_residual')))
    handles.append(model.norm.register_forward_hook(mk_hook('final_norm')))
    # Capture only the last logit separately; full logits are 49k x T.
    try:
        with torch.inference_mode(), inference_context(device, precision, torch):
            logits, _ = model(tokens)
            last = logits[0, prefix_len - 1].detach().float().cpu().clone()
        return snapshots, last
    finally:
        for h in handles:
            h.remove()


def compare_snapshots(left, right, *, atol=5e-3, rtol=1e-3):
    """Absolute numerical differences plus diagnostic elementwise tolerance.

    Layer-level tolerance is *diagnostic*, not a proof of causal correctness.
    """
    if left.keys() != right.keys():
        raise ValueError('Layer snapshots missing or extra between conditions')
    results = []
    first_differing_layer = None
    for key, a in left.items():
        b = right[key]
        if a.shape != b.shape:
            raise ValueError(f'Snapshot shape changed at {key}')
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError(f'Nonfinite layer outputs at {key}')
        delta = (a - b).abs()
        mismatched = int((delta > (atol + rtol * b.abs())).sum().item())
        result = {
            'module': key,
            'max_abs': float(delta.max().item()),
            'mean_abs': float(delta.mean().item()),
            'out_abs_max': float(a.abs().max().item()),
            'elements_above_tolerance': mismatched,
            'elements_total': a.numel(),
        }
        results.append(result)
        if first_differing_layer is None and mismatched:
            first_differing_layer = key
    return {'first_module_over_tolerance': first_differing_layer, 'modules': results}


def suffix_diagnosis(model, ids, *, device, precision, inference_context, compare_fp32=True):
    """Strict suffix experiment with per-layer deltas and optional FP32 cross-check."""
    diag = {}
    precisions = [precision] + (['fp32'] if compare_fp32 and precision != 'fp32' else [])
    for prec in precisions:
        try:
            a, last_a = capture_prefix(model, ids, device=device, precision=prec,
                                       suffix_token=0, inference_context=inference_context)
            b, last_b = capture_prefix(model, ids, device=device, precision=prec,
                                       suffix_token=151, inference_context=inference_context)
            trace = compare_snapshots(a, b)
            d = (last_a - last_b).abs()
            bad = d > (5e-3 + 1e-3 * last_b.abs())
            diag[prec] = {
                'status': 'TRACE_CAPTURED',
                'last_logit_max_abs': float(d.max().item()),
                'last_logit_mean_abs': float(d.mean().item()),
                'last_logit_elements_above_tolerance': int(bad.sum().item()),
                'last_logit_count': d.numel(),
                'last_argmax_original': int(last_a.argmax().item()),
                'last_argmax_changed_suffix': int(last_b.argmax().item()),
                **trace,
            }
        except Exception as exc:
            diag[prec] = {'status': 'DIAG_FAILED', 'exception': f'{type(exc).__name__}: {exc}'}
    return diag
