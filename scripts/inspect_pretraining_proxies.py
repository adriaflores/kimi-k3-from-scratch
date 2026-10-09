#!/usr/bin/env python3
"""Report exact parameter counts and approximate model FLOPs for proxy YAMLs.

Uses the parameter-identical chunkwise KDA on meta, without importing FLA.
FLOPs are an arithmetic estimate, not a profiler result or GPU cost prediction.
See docs/notes/pretraining_proxy_family.md for assumptions and exclusions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_config import (
    component_breakdown,
    count_model_active_params,
    module_params,
)
from src.kimi_k3.model import KimiK3, KimiK3Config
from src.kimi_k3.training.parameter_groups import build_parameter_groups
from src.kimi_k3.utils import load_config


def approximate_flops(config: KimiK3Config, seq_len: int) -> dict:
    """Dominant arithmetic per token; one multiply-add counts as two FLOPs.

    KDA core uses the dense WY reference's dominant products, including an
    approximate triangular solve. MLA counts full square attention products.
    Backward is approximated as twice forward; recomputation is excluded.
    QK observation is a separate no-grad dense upper bound, not tripled.
    """
    if not 1 <= seq_len <= config.max_seq_len:
        raise ValueError("seq_len must be in [1, max_seq_len]")
    c, h, d = config.hidden_size, config.kda_num_heads, config.kda_head_dim
    p = h * d
    kda_weights = 3 * c * p + c * h + c * d + d * p + p * c
    kda_weights += c * p if config.kda_full_rank_output_gate else c * d + d * p
    mla_h = config.mla_num_heads
    q, kv = config.mla_q_lora_rank, config.mla_kv_lora_rank
    dk, ds, dv = config.mla_qk_nope_head_dim, config.mla_qk_shared_head_dim, config.mla_v_head_dim
    mla_weights = c * q + c * (kv + ds) + q * mla_h * (dk + ds) + kv * mla_h * (dk + dv) + mla_h * dv * c
    if config.mla_output_gate:
        mla_weights += c * mla_h * dv
    linear_weights = config.vocab_size * c  # One tied output projection; input is a lookup.
    for index, layer_type in enumerate(config.layer_types):
        linear_weights += kda_weights if layer_type == "kda" else mla_weights
        if index < config.num_dense_layers:
            linear_weights += 3 * c * config.dense_intermediate_size
        else:
            latent, intermediate = config.latent_moe_dim, config.expert_intermediate_size
            linear_weights += (
                c * config.num_experts + 2 * c * latent
                + 3 * c * config.num_shared_experts * intermediate
                + 3 * config.num_experts_per_token * latent * intermediate
            )
    nkda = config.layer_types.count("kda")
    nmla = config.layer_types.count("mla")
    linear_forward = 2 * linear_weights
    conv_forward = nkda * 2 * 3 * p * config.kda_short_conv_kernel_size
    # Whole chunks, including the fused kernel's padded final chunk.
    chunk = config.kda_chunk_size
    padded_tokens = ((seq_len + chunk - 1) // chunk) * chunk
    kda_forward = nkda * h * (6 * d * d + 8 * chunk * d + chunk * chunk) * padded_tokens / seq_len
    mla_forward = nmla * 2 * mla_h * seq_len * (dk + ds + dv)
    forward = linear_forward + conv_forward + kda_forward + mla_forward
    observer_upper_bound = nmla * 2 * mla_h * seq_len * (dk + ds)
    return {
        "seq_len": seq_len,
        "linear_forward": linear_forward,
        "conv_forward": conv_forward,
        "kda_wy_core_forward_approx": kda_forward,
        "mla_attention_forward_dense": mla_forward,
        "forward_approx": forward,
        "forward_backward_approx": 3 * forward,
        "qk_observer_forward_upper_bound": observer_upper_bound,
        "forward_backward_plus_qk_observer_upper_bound": 3 * forward + observer_upper_bound,
    }


def inspect(path: Path, seq_lens: list[int]) -> dict:
    config = load_config(path, KimiK3Config)
    with torch.device("meta"):
        model = KimiK3(replace(config, kda_implementation="chunkwise"))
    total, active = module_params(model), count_model_active_params(model)
    components, active_components = component_breakdown(model, config)
    groups = build_parameter_groups(model, weight_decay=0.1)
    algorithm_counts = {
        algorithm: sum(parameter.numel() for group in groups if group["algorithm"] == algorithm for parameter in group["params"])
        for algorithm in ("muon", "adamw")
    }
    if sum(algorithm_counts.values()) != total:
        raise RuntimeError("Optimizer groups do not cover the model exactly")
    embedding = components["Tied embedding / LM head"]
    return {
        "config": path.as_posix(),
        "configured_kda_backend": config.kda_implementation,
        "counting_kda_backend": "chunkwise/meta",
        "hidden_size": config.hidden_size,
        "layers": config.num_layers,
        "kda_layers": config.layer_types.count("kda"),
        "mla_layers": config.layer_types.count("mla"),
        "kda_heads": config.kda_num_heads,
        "mla_heads": config.mla_num_heads,
        "total_parameters": total,
        "active_parameters_per_token": active,
        "non_embedding_parameters": total - embedding,
        "active_non_embedding_parameters": active - embedding,
        "components_total": dict(components),
        "components_active": active_components,
        "optimizer_parameters": algorithm_counts,
        "parameter_bytes_fp32": 4 * total,
        # All gradients and states allocated: one Muon momentum or two AdamW moments.
        "persistent_training_bytes_upper_bound": 12 * algorithm_counts["muon"] + 16 * algorithm_counts["adamw"],
        "flops_per_token": [approximate_flops(config, length) for length in seq_lens],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+", help="Model YAML paths")
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128])
    parser.add_argument("--output", type=Path, help="Also write the full report as JSON")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "torch_version": str(torch.__version__),
        "flops_convention": "2 FLOPs/MAC; model fwd+bwd approx = 3*fwd; QK observer separate",
        "exclusions": "pointwise/norm/softmax/quantile/routing/sorting/copies/optimizer/recompute/compilation/evaluation",
        "memory_exclusions": "activations, temporary stacks/workspaces, allocator, CUDA runtime; not measured VRAM",
        "models": [inspect(path, args.seq_len) for path in args.configs],
    }
    print(f"{'Config':<42} {'Total M':>10} {'Active M':>10} {'T':>6} {'Train GF/tok*':>15}")
    for model in report["models"]:
        for cost in model["flops_per_token"]:
            print(f"{Path(model['config']).name:<42} {model['total_parameters'] / 1e6:10.3f} "
                  f"{model['active_parameters_per_token'] / 1e6:10.3f} {cost['seq_len']:6d} "
                  f"{cost['forward_backward_approx'] / 1e9:15.4f}")
    print("* Approximate model arithmetic; optimizer and QK observer excluded from this column.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
