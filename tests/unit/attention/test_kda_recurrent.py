import pytest
import torch

from kimi_k3.attention.kda.recurrent import KDARecurrent


def test_output_shape(kda_config):
	model = KDARecurrent(kda_config)
	x = torch.randn(2, 7, kda_config.hidden_size)

	out = model(x)

	assert out.shape == (2, 7, kda_config.hidden_size)


def test_output_dtype_matches_model_dtype(kda_config):
	model = KDARecurrent(kda_config).to(torch.bfloat16)
	x = torch.randn(2, 5, kda_config.hidden_size, dtype=torch.bfloat16)

	out = model(x)

	assert out.dtype == torch.bfloat16


def test_forward_is_finite(kda_config):
	model = KDARecurrent(kda_config)
	x = torch.randn(3, 9, kda_config.hidden_size)

	out = model(x)

	assert torch.isfinite(out).all()


def test_sequence_length_one(kda_config):
	model = KDARecurrent(kda_config)
	x = torch.randn(2, 1, kda_config.hidden_size)

	out = model(x)

	assert out.shape == (2, 1, kda_config.hidden_size)
	assert torch.isfinite(out).all()


def test_backward_produces_finite_gradients(kda_config):
	model = KDARecurrent(kda_config)
	x = torch.randn(2, 6, kda_config.hidden_size)

	out = model(x)
	out.square().mean().backward()

	params_with_grad = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
	assert params_with_grad, "expected at least one parameter to receive a gradient"
	for name, p in params_with_grad:
		assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_causality_future_positions_do_not_affect_past_outputs(kda_config):
	model = KDARecurrent(kda_config)
	model.eval()

	B, T, C = 1, 8, kda_config.hidden_size
	t = 4

	x_a = torch.randn(B, T, C)
	x_b = x_a.clone()
	x_b[:, t + 1:, :] = torch.randn(B, T - (t + 1), C)

	with torch.no_grad():
		out_a = model(x_a)
		out_b = model(x_b)

	torch.testing.assert_close(out_a[:, :t + 1], out_b[:, :t + 1])


@pytest.mark.parametrize("full_rank_output_gate", [True, False])
def test_output_gate_variants_forward_and_backward(kda_config_factory, full_rank_output_gate):
	config = kda_config_factory(kda_full_rank_output_gate=full_rank_output_gate)
	model = KDARecurrent(config)
	x = torch.randn(2, 6, config.hidden_size)

	out = model(x)
	assert out.shape == (2, 6, config.hidden_size)
	assert torch.isfinite(out).all()

	out.square().mean().backward()
	params_with_grad = [p for p in model.parameters() if p.grad is not None]
	assert params_with_grad
	assert all(torch.isfinite(p.grad).all() for p in params_with_grad)
