import json

import pytest
import torch
from torch import nn

from kimi_k3.moe.stable_latent_moe import StableLatentMoE as SourceMoE
from kimi_k3.training.moe_metrics import MoEDiagnostics

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
	not torch.cuda.is_available(), reason="CUDA is unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
def test_loads_use_actual_biased_top_k_and_exclude_eval(kda_config_factory, device):
	config = kda_config_factory(num_experts=4, num_experts_per_token=2)
	moe = SourceMoE(config).to(device)
	x = torch.zeros(1, 2, config.hidden_size, device=device)
	x[..., :4] = torch.tensor([[4.0, 3.0, 0.0, -2.0], [-2.0, 0.0, 3.0, 4.0]], device=device)
	with torch.no_grad():
		moe.router.weight.zero_()
		moe.router.weight[:, :4] = torch.eye(4, device=device)
	keys = set(moe.state_dict())
	with MoEDiagnostics(moe) as diagnostics, torch.no_grad(), torch.autocast(
		device, dtype=torch.bfloat16, enabled=device == "cuda",
	):
		moe(x)  # Unbiased selections: {0,1}, {2,3}.
		after_train = moe.quantile_balancer.biases.clone()
		moe.eval()
		moe(x[:, :1])
		torch.testing.assert_close(moe.quantile_balancer.biases, after_train, rtol=0, atol=0)
		moe.train()
		moe.quantile_balancer.biases.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0], device=device))
		moe(x)  # Biased selections: {0,3}, {2,3}; the (k+1)th expert is excluded.
		latest_biases = moe.quantile_balancer.biases.clone()
		record = diagnostics.snapshot()["<root>"]
		assert record["expert_counts"] == [2, 1, 2, 3]
		assert record["tokens"] == 4
		assert record["max_load_over_mean"] == 1.5
		assert record["zero_load_experts"] == 0
		assert record["qb_biases"] == latest_biases.cpu().tolist()
		torch.testing.assert_close(moe.quantile_balancer.biases, latest_biases, rtol=0, atol=0)
		assert set(moe.state_dict()) == keys
		empty = diagnostics.snapshot()["<root>"]
		assert empty["tokens"] == 0 and empty["expert_counts"] == [0, 0, 0, 0]
		assert empty["max_load_over_mean"] is None and empty["zero_load_experts"] is None
		moe.quantile_balancer.biases.zero_()
		moe(x[:, :1])
		final = diagnostics.snapshot()["<root>"]
		assert final["expert_counts"] == [1, 1, 0, 0]
		assert final["max_load_over_mean"] == 2.0 and final["zero_load_experts"] == 2
	assert moe._expert_load_counts is None


def test_disabled_collection_and_exception_cleanup(kda_config_factory):
	model = nn.Sequential(SourceMoE(kda_config_factory()))
	with MoEDiagnostics(model, enabled=False) as diagnostics:
		model(torch.randn(1, 2, 16))
		assert diagnostics.snapshot() == {}
		assert model[0]._expert_load_counts is None
	with pytest.raises(RuntimeError, match="training failed"), MoEDiagnostics(model) as diagnostics:
		model(torch.randn(1, 2, 16))
		assert diagnostics.snapshot()["0"]["tokens"] == 2
		raise RuntimeError("training failed")
	assert model[0]._expert_load_counts is None
	# A subsequent invocation starts clean, with no saved diagnostic state.
	with MoEDiagnostics(model) as diagnostics:
		assert diagnostics.snapshot()["0"]["tokens"] == 0


def test_nonfinite_bias_snapshot_is_valid_json_and_does_not_mutate_qb(kda_config_factory):
	moe = SourceMoE(kda_config_factory())
	with torch.no_grad():
		moe.quantile_balancer.biases.copy_(torch.tensor([float("nan"), float("inf")]))
	with MoEDiagnostics(moe) as diagnostics:
		record = diagnostics.snapshot()["<root>"]
		assert record["qb_biases"] == [None, None]
		json.dumps(record, allow_nan=False)
	assert torch.isnan(moe.quantile_balancer.biases[0])
	assert torch.isinf(moe.quantile_balancer.biases[1])
