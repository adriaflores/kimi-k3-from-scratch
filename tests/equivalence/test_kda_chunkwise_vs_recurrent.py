import pytest
import torch

from kimi_k3.attention.kda.chunkwise import KDAChunkwise
from kimi_k3.attention.kda.recurrent import KDARecurrent

CHUNK_SIZES = [1, 2, 4, 8]
OUTPUT_GATES = [True, False]


def _make_pair(kda_config_factory, chunk_size, full_rank_output_gate=True):
	config = kda_config_factory(kda_full_rank_output_gate=full_rank_output_gate, kda_chunk_size=chunk_size)
	recurrent = KDARecurrent(config)
	chunkwise = KDAChunkwise(config)
	chunkwise.load_state_dict(recurrent.state_dict())
	return config, recurrent, chunkwise


def _expected_grad_params(full_rank_output_gate):
	names = ["QKV.weight", "Wb.weight", "Wad.weight", "Wau.weight", "A_log", "dt_bias", "Wo.weight"]
	if full_rank_output_gate:
		names.append("Wg.weight")
	else:
		names += ["Wgd.weight", "Wgu.weight"]
	return names


@pytest.mark.parametrize("full_rank_output_gate", OUTPUT_GATES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_forward_matches_recurrent(kda_config_factory, chunk_size, full_rank_output_gate):
	config, recurrent, chunkwise = _make_pair(kda_config_factory, chunk_size, full_rank_output_gate)
	x = torch.randn(2, 16, config.hidden_size)

	with torch.no_grad():
		y_recurrent = recurrent(x)
		y_chunkwise = chunkwise(x)

	torch.testing.assert_close(y_chunkwise, y_recurrent)


@pytest.mark.parametrize("full_rank_output_gate", OUTPUT_GATES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_single_chunk_matches_recurrent(kda_config_factory, chunk_size, full_rank_output_gate):
	config, recurrent, chunkwise = _make_pair(kda_config_factory, chunk_size, full_rank_output_gate)
	x = torch.randn(2, chunk_size, config.hidden_size)

	with torch.no_grad():
		y_recurrent = recurrent(x)
		y_chunkwise = chunkwise(x)

	torch.testing.assert_close(y_chunkwise, y_recurrent)


@pytest.mark.parametrize("full_rank_output_gate", OUTPUT_GATES)
def test_many_chunks_matches_recurrent(kda_config_factory, full_rank_output_gate):
	# T=16, C=4 forces three S_out -> S_in,next transitions.
	config, recurrent, chunkwise = _make_pair(kda_config_factory, 4, full_rank_output_gate)
	x = torch.randn(2, 16, config.hidden_size)

	with torch.no_grad():
		y_recurrent = recurrent(x)
		y_chunkwise = chunkwise(x)

	torch.testing.assert_close(y_chunkwise, y_recurrent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_shape_and_dtype(kda_config_factory, dtype):
	config, _, chunkwise = _make_pair(kda_config_factory, 4)
	chunkwise = chunkwise.to(dtype)
	x = torch.randn(2, 8, config.hidden_size, dtype=dtype)

	out = chunkwise(x)

	assert out.shape == (2, 8, config.hidden_size)
	assert out.dtype == dtype


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_forward_is_finite(kda_config_factory, chunk_size):
	config, _, chunkwise = _make_pair(kda_config_factory, chunk_size)
	# Large inputs push the gates toward the lower bound, stressing exp(g_t - g_j).
	x = 10 * torch.randn(3, 16, config.hidden_size)

	out = chunkwise(x)

	assert torch.isfinite(out).all()


@pytest.mark.parametrize("full_rank_output_gate", OUTPUT_GATES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_backward_matches_recurrent(kda_config_factory, chunk_size, full_rank_output_gate):
	config, recurrent, chunkwise = _make_pair(kda_config_factory, chunk_size, full_rank_output_gate)
	x = torch.randn(2, 16, config.hidden_size)
	x_recurrent = x.clone().requires_grad_(True)
	x_chunkwise = x.clone().requires_grad_(True)

	recurrent(x_recurrent).square().mean().backward()
	chunkwise(x_chunkwise).square().mean().backward()

	torch.testing.assert_close(x_chunkwise.grad, x_recurrent.grad)
	recurrent_params = dict(recurrent.named_parameters())
	chunkwise_params = dict(chunkwise.named_parameters())
	for name in _expected_grad_params(full_rank_output_gate):
		torch.testing.assert_close(
			chunkwise_params[name].grad, recurrent_params[name].grad, msg=lambda m, name=name: f"{name}: {m}"
		)


@pytest.mark.parametrize("full_rank_output_gate", OUTPUT_GATES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_backward_produces_finite_gradients(kda_config_factory, chunk_size, full_rank_output_gate):
	config, _, chunkwise = _make_pair(kda_config_factory, chunk_size, full_rank_output_gate)
	x = torch.randn(2, 16, config.hidden_size, requires_grad=True)

	chunkwise(x).square().mean().backward()

	assert torch.isfinite(x.grad).all()
	params = dict(chunkwise.named_parameters())
	for name in _expected_grad_params(full_rank_output_gate):
		assert params[name].grad is not None, f"no gradient for {name}"
		assert torch.isfinite(params[name].grad).all(), f"non-finite gradient for {name}"


@pytest.mark.parametrize("t", [0, 3, 4, 9, 14])
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_causality_future_positions_do_not_affect_past_outputs(kda_config_factory, chunk_size, t):
	# t covers the first token, both sides of chunk boundaries (3|4 for C=4), and mid-chunk positions.
	config, _, chunkwise = _make_pair(kda_config_factory, chunk_size)
	chunkwise.eval()

	B, T, C = 1, 16, config.hidden_size
	x_a = torch.randn(B, T, C)
	x_b = x_a.clone()
	x_b[:, t + 1:, :] = torch.randn(B, T - (t + 1), C)

	with torch.no_grad():
		out_a = chunkwise(x_a)
		out_b = chunkwise(x_b)

	torch.testing.assert_close(out_a[:, :t + 1], out_b[:, :t + 1])


def test_sequence_length_not_divisible_by_chunk_size_raises(kda_config_factory):
	config, _, chunkwise = _make_pair(kda_config_factory, 4)
	x = torch.randn(2, 10, config.hidden_size)

	with pytest.raises(AssertionError, match="must be divisible by chunk_size"):
		chunkwise(x)
