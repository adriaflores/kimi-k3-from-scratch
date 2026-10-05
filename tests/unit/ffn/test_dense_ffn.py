import pytest
import torch

from kimi_k3.ffn.dense_ffn import DenseFFN, SituGLU


def _make_situ(config):
	return SituGLU(beta=config.situ_beta, linear_beta=config.situ_linear_beta)


def _make_ffn(config):
	return DenseFFN(config.hidden_size, config.dense_intermediate_size, _make_situ(config))


def _reference_situ(x, beta, linear_beta):
	"""SiTU-GLU written out in FP32: (beta * tanh(g / beta) * sigmoid(g)) * (linear_beta * tanh(v / linear_beta))."""
	gate, values = x.float().chunk(2, dim=-1)
	gate_branch = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
	value_branch = linear_beta * torch.tanh(values / linear_beta)
	return gate_branch * value_branch


@pytest.mark.parametrize("B, T", [(1, 1), (2, 7), (4, 16)])
def test_dense_ffn_output_shape(kda_config, B, T):
	ffn = _make_ffn(kda_config)
	x = torch.randn(B, T, kda_config.hidden_size)

	out = ffn(x)

	assert out.shape == (B, T, kda_config.hidden_size)


@pytest.mark.parametrize("B, T", [(1, 1), (2, 7), (4, 16)])
def test_situ_glu_halves_last_dimension(kda_config, B, T):
	situ = _make_situ(kda_config)
	I = kda_config.dense_intermediate_size
	x = torch.randn(B, T, 2 * I)

	out = situ(x)

	assert out.shape == (B, T, I)


def test_situ_glu_forward_is_finite(kda_config):
	situ = _make_situ(kda_config)
	x = torch.randn(3, 9, 2 * kda_config.dense_intermediate_size)

	out = situ(x)

	assert torch.isfinite(out).all()


def test_dense_ffn_forward_is_finite(kda_config):
	ffn = _make_ffn(kda_config)
	x = torch.randn(3, 9, kda_config.hidden_size)

	out = ffn(x)

	assert torch.isfinite(out).all()


def test_dense_ffn_backward_produces_finite_gradients(kda_config):
	ffn = _make_ffn(kda_config)
	x = torch.randn(2, 6, kda_config.hidden_size, requires_grad=True)

	ffn(x).square().mean().backward()

	assert x.grad is not None
	assert torch.isfinite(x.grad).all()
	for name in ["gate_up_proj.weight", "down_proj.weight"]:
		grad = dict(ffn.named_parameters())[name].grad
		assert grad is not None, f"no gradient for {name}"
		assert torch.isfinite(grad).all(), f"non-finite gradient for {name}"
		assert grad.abs().sum() > 0, f"zero gradient for {name}"


def test_situ_glu_bf16_round_trip(kda_config):
	situ = _make_situ(kda_config)
	x = torch.randn(2, 5, 2 * kda_config.dense_intermediate_size).to(torch.bfloat16)

	out = situ(x)

	assert out.dtype == torch.bfloat16
	assert torch.isfinite(out).all()
	# Computing internally in FP32 and casting once at the end gives exactly the FP32 result rounded to BF16.
	expected = _reference_situ(x, kda_config.situ_beta, kda_config.situ_linear_beta).to(torch.bfloat16)
	torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta, linear_beta", [(1.0, 1.0), (0.5, 2.0)])
def test_situ_glu_matches_reference(beta, linear_beta):
	situ = SituGLU(beta=beta, linear_beta=linear_beta)
	x = torch.randn(2, 5, 12)

	torch.testing.assert_close(situ(x), _reference_situ(x, beta, linear_beta))


def test_dense_ffn_matches_reference(kda_config):
	ffn = _make_ffn(kda_config)
	x = torch.randn(2, 4, kda_config.hidden_size)

	with torch.no_grad():
		gate_up = x @ ffn.gate_up_proj.weight.T
		activated = _reference_situ(gate_up, kda_config.situ_beta, kda_config.situ_linear_beta)
		expected = activated @ ffn.down_proj.weight.T

		out = ffn(x)

	torch.testing.assert_close(out, expected)
