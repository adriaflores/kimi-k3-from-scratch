import pytest
import torch
from torch import nn

from kimi_k3.utils.rmsn import MixedPrecisionRMSNorm

DEVICES = ["cpu", pytest.param(
	"cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))]


def _make_pair(device, hidden_size=16):
	reference = nn.RMSNorm(hidden_size, eps=1e-6).to(device)
	nn.init.normal_(reference.weight)
	mixed = MixedPrecisionRMSNorm(hidden_size, eps=1e-6).to(device)
	mixed.load_state_dict(reference.state_dict())
	return reference, mixed


def test_state_dict_matches_nn_rmsnorm():
	reference, mixed = _make_pair("cpu")

	assert mixed.state_dict().keys() == reference.state_dict().keys()
	assert mixed.weight.dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_matches_nn_rmsnorm_exactly(device):
	reference, mixed = _make_pair(device)
	x = torch.randn(2, 8, 16, device=device)
	x_reference = x.clone().requires_grad_()
	x_mixed = x.clone().requires_grad_()

	y_reference = reference(x_reference)
	y_mixed = mixed(x_mixed)
	y_reference.square().sum().backward()
	y_mixed.square().sum().backward()

	torch.testing.assert_close(y_mixed, y_reference, rtol=0, atol=0)
	torch.testing.assert_close(x_mixed.grad, x_reference.grad, rtol=0, atol=0)
	torch.testing.assert_close(mixed.weight.grad, reference.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_input_keeps_bf16_output_and_fp32_weight_grad(device):
	reference, mixed = _make_pair(device)
	x = torch.randn(2, 8, 16, device=device)
	x_mixed = x.to(torch.bfloat16).requires_grad_()

	y_reference = reference(x)
	y_mixed = mixed(x_mixed)
	y_mixed.float().square().sum().backward()

	assert y_mixed.dtype == torch.bfloat16
	assert mixed.weight.dtype == torch.float32
	assert mixed.weight.grad.dtype == torch.float32
	assert x_mixed.grad.dtype == torch.bfloat16
	torch.testing.assert_close(y_mixed.float(), y_reference.detach(), rtol=2e-2, atol=2e-2)
