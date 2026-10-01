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


def test_adamw_groups_cover_tied_weights_once_and_exclude_1d_decay(kda_config_factory):
	model = KimiK3(kda_config_factory())
	config = TrainConfig(weight_decay=0.1)
	optimizer = build_optimizer(model, config)
	assert model.embed_tokens.weight is model.lm_head.weight
	flattened = [parameter for group in optimizer.param_groups for parameter in group["params"]]
	assert len(flattened) == len({id(parameter) for parameter in flattened})
	assert {id(parameter) for parameter in flattened} == {id(parameter) for parameter in model.parameters()}
	for group in optimizer.param_groups:
		assert group["betas"] == (0.9, 0.95)
		assert group["eps"] == 1e-8
		for parameter in group["params"]:
			assert group["weight_decay"] == (0.1 if parameter.ndim >= 2 else 0)


@pytest.mark.parametrize("name,steps,accumulation,overfit", [
	("smoke", 20, 1, False), ("overfit", 2000, 1, True), ("stability", 1000, 4, False)])
def test_preset_loads_shared_yaml_schema_and_cli_overrides(name, steps, accumulation, overfit):
	path = REPO_ROOT / f"configs/training/{name}.yaml"
	config = load_config(path, TrainConfig)
	assert (config.max_steps, config.gradient_accumulation_steps, config.overfit_batch) == (
		steps, accumulation, overfit)
	assert config.batch_size * config.seq_len * accumulation == 128 * accumulation
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
])
def test_invalid_recipe_is_rejected(kwargs):
	with pytest.raises(ValueError):
		TrainConfig(**kwargs)


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
