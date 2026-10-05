import pytest
import torch

pytest.importorskip("fla.ops.kda")

from kimi_k3.attention.kda.chunkwise import KDAChunkwise
from kimi_k3.attention.kda.fused_chunk import FusedKDAChunkwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fla Triton kernels need a CUDA device")

# fla only accepts chunk_size 32/64 and lower_bound in [-5, 0); chunk_size=32 hits a
# misaligned-address error on triton-windows, so we compare at 64.
CHUNK_SIZE = 64
HEAD_DIMS = [64, 128]
OUTPUT_GATES = [True, False]

# The fla kernels use TF32 tensor-core dots, so FP32 agreement is ~1e-3, not bitwise.
FP32_RTOL = 5e-3
BF16_RTOL = 3e-2
# dA_log[h] sums da * (a + dt_bias) over every (b, t, d); those terms cancel heavily
# (~177x at head_dim=128 for this seed), amplifying the ~1e-3 per-element error to ~4%.
A_LOG_GRAD_RTOL = 1e-1


def _make_pair(kda_config_factory, head_dim, full_rank_output_gate=True):
	config = kda_config_factory(
		hidden_size=64,
		kda_num_heads=2,
		kda_head_dim=head_dim,
		kda_chunk_size=CHUNK_SIZE,
		kda_gate_lower_bound=-5.0,
		kda_full_rank_output_gate=full_rank_output_gate,
	)
	reference = KDAChunkwise(config).cuda()
	fused = FusedKDAChunkwise(config).cuda()
	fused.load_state_dict(reference.state_dict())
	return config, reference, fused


def _rel_error(actual, expected):
	return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("full_rank_output_gate", OUTPUT_GATES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_fused_forward_matches_chunkwise(kda_config_factory, head_dim, full_rank_output_gate):
	config, reference, fused = _make_pair(kda_config_factory, head_dim, full_rank_output_gate)
	# Two chunks, so the inter-chunk state carry is exercised.
	x = torch.randn(2, 2 * CHUNK_SIZE, config.hidden_size, device="cuda")

	with torch.no_grad():
		y_reference = reference(x)
		y_fused = fused(x)

	assert y_fused.shape == y_reference.shape
	assert _rel_error(y_fused, y_reference) < FP32_RTOL


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_fused_backward_matches_chunkwise(kda_config_factory, head_dim):
	config, reference, fused = _make_pair(kda_config_factory, head_dim)
	x = torch.randn(2, 2 * CHUNK_SIZE, config.hidden_size, device="cuda")
	x_reference = x.clone().requires_grad_()
	x_fused = x.clone().requires_grad_()
	grad_out = torch.randn(2, 2 * CHUNK_SIZE, config.hidden_size, device="cuda")

	reference(x_reference).backward(grad_out)
	fused(x_fused).backward(grad_out)

	assert _rel_error(x_fused.grad, x_reference.grad) < FP32_RTOL
	fused_params = dict(fused.named_parameters())
	for name, p_reference in reference.named_parameters():
		p_fused = fused_params[name]
		assert p_fused.grad is not None, name
		rtol = A_LOG_GRAD_RTOL if name == "A_log" else FP32_RTOL
		assert _rel_error(p_fused.grad, p_reference.grad) < rtol, name


def test_fused_bf16_autocast_matches_chunkwise(kda_config_factory):
	config, reference, fused = _make_pair(kda_config_factory, 64)
	x = torch.randn(2, 2 * CHUNK_SIZE, config.hidden_size, device="cuda")

	with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
		y_reference = reference(x)
		y_fused = fused(x)

	assert torch.isfinite(y_fused).all()
	assert _rel_error(y_fused, y_reference) < BF16_RTOL
