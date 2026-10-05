#!/usr/bin/env python3
"""
Inspect a Kimi K3 model YAML and report exact total/active parameter counts.

Run from the repository root:

	uv run python scripts/inspect_config.py configs/model/kimi_k3_300m.yaml

Optional:

	uv run python scripts/inspect_config.py configs/model/kimi_k3_300m.yaml --by-layer
	uv run python scripts/inspect_config.py configs/model/kimi_k3_300m.yaml --device cpu

The default `meta` device instantiates the full module graph without allocating
storage for model parameters, so even very large reference configs can be
inspected safely.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kimi_k3.utils import load_config

from src.kimi_k3.model import KimiK3, KimiK3Config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Instantiate a Kimi K3 config and inspect its parameter budget."
	)
	parser.add_argument("config", type=Path, help="Path to a model YAML.")
	parser.add_argument(
		"--device",
		choices=("meta", "cpu"),
		default="meta",
		help="Instantiation device. Default: meta (no parameter storage allocation).",
	)
	parser.add_argument(
		"--by-layer",
		action="store_true",
		help="Show total and active parameters for every decoder layer.",
	)
	parser.add_argument(
		"--show-config",
		action="store_true",
		help="Print the normalized model config before the report.",
	)
	return parser.parse_args()


def unique_numel(parameters) -> int:
	seen: set[int] = set()
	total = 0

	for p in parameters:
		pid = id(p)
		if pid not in seen:
			seen.add(pid)
			total += p.numel()

	return total


def module_params(module: nn.Module) -> int:
	return unique_numel(module.parameters())


def routed_expert_params(ffn: nn.Module) -> tuple[int, int]:
	"""
	Return (all_routed_expert_params, active_routed_expert_params).

	The current StableLatentMoE stores routed experts in `ffn.experts` as a
	ModuleList of equal-sized DenseFFNs.
	"""
	experts = getattr(ffn, "experts", None)
	top_k = getattr(ffn, "top_k", None)

	if not isinstance(experts, nn.ModuleList) or top_k is None:
		return 0, 0
	if len(experts) == 0:
		return 0, 0

	sizes = [module_params(expert) for expert in experts]
	if len(set(sizes)) != 1:
		raise RuntimeError(
			"Routed experts have unequal parameter counts; "
			"the active-parameter estimate needs updating."
		)

	per_expert = sizes[0]
	return len(experts) * per_expert, top_k * per_expert


def active_module_params(module: nn.Module) -> int:
	total = module_params(module)
	inactive = 0

	for child in module.modules():
		all_experts, active_experts = routed_expert_params(child)

		if all_experts:
			inactive += all_experts - active_experts

	return total - inactive


def count_model_active_params(model: KimiK3) -> int:
	total = module_params(model)
	inactive = 0

	for layer in model.layers:
		all_experts, active_experts = routed_expert_params(layer.ffn)
		inactive += all_experts - active_experts

	return total - inactive


def classify_parameter(name: str, config: KimiK3Config) -> str:
	if name.startswith("embed_tokens."):
		return "Tied embedding / LM head"

	if name.startswith("layers."):
		parts = name.split(".")
		layer_idx = int(parts[1])
		submodule = parts[2]

		if submodule == "attn":
			return "KDA" if config.layer_types[layer_idx] == "kda" else "Gated MLA"

		if submodule == "ffn":
			return "Dense FFN" if layer_idx < config.num_dense_layers else "Stable LatentMoE"

		return "Decoder norms + AttnRes"

	if (
		name.startswith("norm.")
		or name.startswith("output_attn_res_norm.")
		or name.startswith("output_attn_res_proj.")
	):
		return "Final norm + AttnRes"

	# With tied weights, lm_head.weight is normally removed as a duplicate by
	# named_parameters(remove_duplicate=True). Keep this case for clarity if
	# the tying implementation changes.
	if name.startswith("lm_head."):
		return "LM head"

	return "Other"


def component_breakdown(model: KimiK3, config: KimiK3Config):
	totals = defaultdict(int)

	for name, p in model.named_parameters(remove_duplicate=True):
		totals[classify_parameter(name, config)] += p.numel()

	# Active count only differs for Stable LatentMoE routed experts.
	active = dict(totals)
	moe_inactive = 0

	for layer in model.layers:
		all_experts, active_experts = routed_expert_params(layer.ffn)
		moe_inactive += all_experts - active_experts

	if "Stable LatentMoE" in active:
		active["Stable LatentMoE"] -= moe_inactive

	return totals, active


def fmt_count(n: int) -> str:
	if n >= 1_000_000_000:
		return f"{n / 1_000_000_000:.3f}B"
	if n >= 1_000_000:
		return f"{n / 1_000_000:.3f}M"
	if n >= 1_000:
		return f"{n / 1_000:.3f}K"
	return str(n)


def fmt_memory(n_bytes: int) -> str:
	if n_bytes >= 1024**3:
		return f"{n_bytes / 1024**3:.2f} GiB"
	return f"{n_bytes / 1024**2:.2f} MiB"


def print_config_summary(config: KimiK3Config) -> None:
	n_kda = sum(t == "kda" for t in config.layer_types)
	n_mla = sum(t == "mla" for t in config.layer_types)

	print("=== Architecture ===")
	print(f"Vocabulary:           {config.vocab_size:,}")
	print(f"Hidden size:          {config.hidden_size:,}")
	print(f"Layers:               {config.num_layers} ({n_kda} KDA / {n_mla} MLA)")
	print(f"Dense / MoE layers:   {config.num_dense_layers} / "
		  f"{config.num_layers - config.num_dense_layers}")
	print(f"Routed experts:       {config.num_experts} (top-{config.num_experts_per_token})")
	print(f"Shared experts:       {config.num_shared_experts}")
	print(f"Max sequence length:  {config.max_seq_len:,}")
	print(f"KDA chunk size:       {config.kda_chunk_size}")
	print(f"KDA implementation:   {config.kda_implementation}")


def print_report(
	model: KimiK3,
	config: KimiK3Config,
	*,
	by_layer: bool,
) -> None:
	total = module_params(model)
	active = count_model_active_params(model)
	trainable = unique_numel(p for p in model.parameters() if p.requires_grad)
	buffers = sum(b.numel() for b in model.buffers())

	print()
	print_config_summary(config)

	print("\n=== Parameter budget ===")
	print(f"Total parameters:       {fmt_count(total):>10}  ({total:,})")
	print(f"Active / token*:        {fmt_count(active):>10}  ({active:,})")
	print(f"Active / total:         {100.0 * active / total:9.2f}%")
	print(f"Trainable parameters:   {fmt_count(trainable):>10}  ({trainable:,})")
	print(f"Buffers:                {fmt_count(buffers):>10}  ({buffers:,})")
	print(
		"* Active = all parameters except routed experts outside top-k; "
		"shared experts and router are included."
	)

	print("\n=== Parameter-only memory ===")
	print(f"FP32:                  {fmt_memory(total * 4):>10}")
	print(f"BF16 / FP16:           {fmt_memory(total * 2):>10}")

	totals, actives = component_breakdown(model, config)

	order = [
		"Tied embedding / LM head",
		"KDA",
		"Gated MLA",
		"Dense FFN",
		"Stable LatentMoE",
		"Decoder norms + AttnRes",
		"Final norm + AttnRes",
		"LM head",
		"Other",
	]

	print("\n=== Component breakdown ===")
	print(f"{'Component':<28} {'Total':>12} {'Active':>12} {'% total':>9}")
	print("-" * 65)

	for name in order:
		if name not in totals:
			continue
		n_total = totals[name]
		n_active = actives.get(name, n_total)
		print(
			f"{name:<28} "
			f"{fmt_count(n_total):>12} "
			f"{fmt_count(n_active):>12} "
			f"{100.0 * n_total / total:8.2f}%"
		)

	if by_layer:
		print("\n=== Decoder layers ===")
		print(f"{'Layer':<8} {'Type':<8} {'FFN':<7} {'Total':>12} {'Active':>12}")
		print("-" * 55)

		for i, layer in enumerate(model.layers):
			layer_type = config.layer_types[i].upper()
			ffn_type = "Dense" if i < config.num_dense_layers else "MoE"
			layer_total = module_params(layer)
			layer_active = active_module_params(layer)

			print(
				f"{i:<8} {layer_type:<8} {ffn_type:<7} "
				f"{fmt_count(layer_total):>12} {fmt_count(layer_active):>12}"
			)


def main() -> None:
	args = parse_args()
	config = load_config(args.config, KimiK3Config)

	if args.show_config:
		print("=== Normalized config ===")
		for field in fields(KimiK3Config):
			print(f"{field.name}: {getattr(config, field.name)}")
		print()

	if args.device == "meta":
		with torch.device("meta"):
			model = KimiK3(config)
	else:
		model = KimiK3(config)

	print(f"Config: {args.config}")
	print_report(model, config, by_layer=args.by_layer)


if __name__ == "__main__":
	main()
