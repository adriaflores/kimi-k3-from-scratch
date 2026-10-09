from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.train import build_train_config, parse_args
from src.kimi_k3.model import KimiK3
from src.kimi_k3.training.trainer import (
	TrainConfig,
	build_optimizer,
	learning_rate_at_step,
	validate_precision,
)
from src.kimi_k3.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_warmup_cosine_boundaries_do_not_depend_on_interruption_length():
	config = TrainConfig(learning_rate=1e-3, lr_schedule="cosine", warmup_steps=2,
						 decay_steps=6, min_lr_ratio=0.1, max_steps=6)
	assert learning_rate_at_step(config, 1) == pytest.approx(5e-4)
	assert learning_rate_at_step(config, 2) == pytest.approx(1e-3)
	assert learning_rate_at_step(config, 4) == pytest.approx(5.5e-4)
	assert learning_rate_at_step(config, 6) == pytest.approx(1e-4)
	assert learning_rate_at_step(config, 10) == pytest.approx(1e-4)
	interrupted = replace(config, max_steps=2)
	assert [learning_rate_at_step(interrupted, step) for step in range(1, 7)] == [
		learning_rate_at_step(config, step) for step in range(1, 7)]
	constant = replace(config, lr_schedule="constant")
	assert learning_rate_at_step(constant, 1) == pytest.approx(5e-4)
	assert learning_rate_at_step(constant, 10) == config.learning_rate
	no_warmup = replace(config, warmup_steps=0)
	assert learning_rate_at_step(no_warmup, 1) == config.learning_rate


def test_hybrid_groups_cover_tied_weights_once_and_exclude_vectors_from_decay(kda_config_factory):
	model = KimiK3(kda_config_factory())
	config = TrainConfig(weight_decay=0.1)
	optimizer = build_optimizer(model, config)
	assert model.embed_tokens.weight is model.lm_head.weight
	flattened = [parameter for group in optimizer.param_groups for parameter in group["params"]]
	assert len(flattened) == len({id(parameter) for parameter in flattened})
	assert {id(parameter) for parameter in flattened} == {id(parameter) for parameter in model.parameters()}
	for group in optimizer.param_groups:
		if group["algorithm"] == "adamw":
			assert group["betas"] == (0.9, 0.95)
			assert group["eps"] == 1e-8
		for parameter in group["params"]:
			if parameter.ndim < 2:
				assert group["algorithm"] == "adamw" and group["weight_decay"] == 0
	assert {group["algorithm"] for group in optimizer.param_groups} == {"muon", "adamw"}


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_adam_fused_reaches_only_auxiliary_groups(kda_config_factory, weight_decay):
	model = KimiK3(kda_config_factory())

	default = build_optimizer(model, TrainConfig(weight_decay=weight_decay))
	unfused = build_optimizer(model, TrainConfig(weight_decay=weight_decay, adam_fused=False))

	assert all(group["fused"] is False for group in unfused.param_groups if group["algorithm"] == "adamw")
	assert all(group["fused"] is True for group in default.param_groups if group["algorithm"] == "adamw")
	assert all("fused" not in group for group in default.param_groups if group["algorithm"] == "muon")


DEVICES = ["cpu", pytest.param(
	"cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
def test_fused_adamw_updates_match_the_default_implementation(kda_config_factory, device):
	torch.manual_seed(0)
	reference = KimiK3(kda_config_factory()).to(device)
	fused = KimiK3(kda_config_factory()).to(device)
	fused.load_state_dict(reference.state_dict())
	config = TrainConfig(learning_rate=1e-2, weight_decay=0.1)
	optimizers = [build_optimizer(reference, replace(config, adam_fused=False)), build_optimizer(fused, config)]
	idx = torch.randint(0, reference.config.vocab_size, (2, 16), device=device)

	for _ in range(3):
		for optimizer in optimizers:
			optimizer.zero_grad(set_to_none=True)
		_, loss = reference(idx, targets=idx)
		loss.backward()
		# Compare execution backends on identical gradients. Small AdamW rounding
		# differences can cross BF16 Newton-Schulz rounding boundaries on later forwards.
		for expected, actual in zip(reference.parameters(), fused.parameters()):
			actual.grad = None if expected.grad is None else expected.grad.clone()
		for optimizer in optimizers:
			optimizer.step()

	for (name, expected), actual in zip(reference.state_dict().items(), fused.state_dict().values()):
		torch.testing.assert_close(actual, expected, msg=name)


def test_adam_fused_cli_flag_overrides_the_recipe():
	base = ["--data-dir", "unused", "--smoke-test"]
	assert build_train_config(parse_args(base)).adam_fused is True
	assert build_train_config(parse_args(base + ["--adam-fused"])).adam_fused is True
	assert build_train_config(parse_args(base + ["--no-adam-fused"])).adam_fused is False


def test_muon_cli_overrides_without_an_optimizer_selector():
	args = parse_args(["--data-dir", "unused", "--smoke-test", "--muon-momentum", "0.9",
		"--muon-ns-steps", "7", "--muon-ns-dtype", "fp32"])
	config = build_train_config(args)
	assert (config.muon_momentum, config.muon_ns_steps, config.muon_ns_dtype) == (0.9, 7, "fp32")
	assert not hasattr(args, "optimizer")


def test_qk_clip_cli_overrides_the_fixed_positive_recipe():
	args = parse_args(["--data-dir", "unused", "--smoke-test", "--qk-clip-threshold", "80",
		"--qk-clip-chunk-size", "64"])
	config = build_train_config(args)
	assert (config.qk_clip_threshold, config.qk_clip_chunk_size) == (80.0, 64)
	assert not hasattr(args, "qk_clip_enabled")


@pytest.mark.parametrize("name,steps,accumulation,overfit", [
	("smoke", 20, 1, False), ("overfit", 2000, 1, True), ("stability", 1000, 4, False),
	("acceptance", 64, 2, True)])
def test_preset_loads_shared_yaml_schema_and_cli_overrides(name, steps, accumulation, overfit):
	path = REPO_ROOT / f"configs/training/{name}.yaml"
	config = load_config(path, TrainConfig)
	assert (config.max_steps, config.gradient_accumulation_steps, config.overfit_batch) == (
		steps, accumulation, overfit)
	assert config.batch_size * config.seq_len * accumulation == 128 * accumulation
	assert (config.qk_clip_threshold, config.qk_clip_chunk_size) == (100.0, 256)
	args = parse_args(["--data-dir", "unused", "--smoke-test", "--training-config", str(path),
					   "--max-steps", "3", "--batch-size", "2", "--no-overfit-batch"])
	overridden = build_train_config(args)
	assert overridden.max_steps == 3
	assert overridden.batch_size == 2
	assert not overridden.overfit_batch
	assert overridden.learning_rate == config.learning_rate
	assert overridden.gradient_accumulation_steps == accumulation


@pytest.mark.parametrize("kwargs", [
	{"gradient_accumulation_steps": 0}, {"precision": "int8"},
	{"lr_schedule": "cosine", "warmup_steps": 100, "decay_steps": 100},
	{"adam_beta2": 1.0}, {"max_grad_norm": float("nan")}, {"min_lr_ratio": -0.1},
	{"muon_momentum": 1.0}, {"muon_momentum": float("nan")}, {"muon_momentum": True},
	{"muon_ns_steps": 0}, {"muon_ns_steps": True}, {"muon_ns_steps": 1.5}, {"muon_ns_dtype": "fp16"},
	{"qk_clip_threshold": 0}, {"qk_clip_threshold": -1}, {"qk_clip_threshold": True},
	{"qk_clip_threshold": float("inf")}, {"qk_clip_threshold": float("nan")},
	{"qk_clip_chunk_size": 0}, {"qk_clip_chunk_size": True}, {"qk_clip_chunk_size": 1.5},
])
def test_invalid_recipe_is_rejected(kwargs):
	with pytest.raises(ValueError):
		TrainConfig(**kwargs)


def test_non_bool_adam_fused_is_rejected():
	with pytest.raises(TypeError, match="adam_fused"):
		TrainConfig(adam_fused="no")


def test_muon_compile_cli_and_config_reach_only_muon_groups(kda_config_factory):
	args = parse_args(["--data-dir", "unused", "--smoke-test", "--muon-compile"])
	config = build_train_config(args)
	assert config.muon_compile is True
	optimizer = build_optimizer(KimiK3(kda_config_factory()), config)
	assert all(group["ns_compile"] is True for group in optimizer.param_groups if group["algorithm"] == "muon")
	assert all("ns_compile" not in group for group in optimizer.param_groups if group["algorithm"] == "adamw")
	args = parse_args(["--data-dir", "unused", "--smoke-test", "--no-muon-compile"])
	assert build_train_config(args).muon_compile is False
	with pytest.raises(TypeError, match="muon_compile"):
		TrainConfig(muon_compile=1)


def test_fp16_cpu_is_rejected_before_running_training():
	with pytest.raises(ValueError, match="requires CUDA"):
		validate_precision(torch.device("cpu"), "fp16")


def test_autocast_preserves_fp32_kda_solve_and_router(kda_config_factory, monkeypatch):
	model = KimiK3(kda_config_factory(num_layers=4, num_dense_layers=1, attn_res_block_size=2,
								   layer_types=("kda", "kda", "kda", "mla")))
	solve = torch.linalg.solve_triangular
	calls = []

	def checked_solve(matrix, rhs, **kwargs):
		assert matrix.dtype == rhs.dtype == torch.float32
		assert not torch.is_autocast_enabled("cpu")
		calls.append("solve")
		return solve(matrix, rhs, **kwargs)

	monkeypatch.setattr(torch.linalg, "solve_triangular", checked_solve)

	def check_router(balancer):
		update = balancer.update

		def checked_update(top_scores, router_scores):
			assert top_scores.dtype == router_scores.dtype == torch.float32
			assert not torch.is_autocast_enabled("cpu")
			calls.append("router")
			return update(top_scores, router_scores)

		monkeypatch.setattr(balancer, "update", checked_update)

	for module in model.modules():
		if hasattr(module, "quantile_balancer"):
			check_router(module.quantile_balancer)
	x = torch.randint(0, 32, (1, 8))
	with torch.autocast("cpu", dtype=torch.bfloat16):
		_, loss = model(x, targets=x)
	loss.backward()
	assert torch.isfinite(loss)
	assert calls.count("solve") == calls.count("router") == 3
	assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
