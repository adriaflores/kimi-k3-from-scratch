import pytest


@pytest.mark.parametrize("num_experts_per_token", [0, 4, 5])
def test_num_experts_per_token_must_leave_room_for_balancer_threshold(kda_config_factory, num_experts_per_token):
	# The QuantileBalancer reads the (k+1)-th routed score, so top-k must be strictly below num_experts.
	with pytest.raises(ValueError, match="num_experts_per_token"):
		kda_config_factory(num_experts=4, num_experts_per_token=num_experts_per_token)


def test_largest_valid_num_experts_per_token_is_accepted(kda_config_factory):
	config = kda_config_factory(num_experts=4, num_experts_per_token=3)

	assert config.num_experts_per_token == 3


def test_unknown_kda_implementation_is_rejected(kda_config_factory):
	with pytest.raises(ValueError, match="kda_implementation"):
		kda_config_factory(kda_implementation="triton")


@pytest.mark.parametrize("chunk_size", [4, 16, 128])
def test_fused_kda_requires_a_supported_chunk_size(kda_config_factory, chunk_size):
	with pytest.raises(ValueError, match="kda_chunk_size"):
		kda_config_factory(kda_implementation="fused", kda_chunk_size=chunk_size, kda_gate_lower_bound=-5.0)


@pytest.mark.parametrize("lower_bound", [-8.0, -5.5, 0.0])
def test_fused_kda_requires_a_safe_gate_lower_bound(kda_config_factory, lower_bound):
	with pytest.raises(ValueError, match="kda_gate_lower_bound"):
		kda_config_factory(kda_implementation="fused", kda_chunk_size=64, kda_gate_lower_bound=lower_bound)


def test_chunkwise_kda_keeps_arbitrary_chunk_size_and_lower_bound(kda_config_factory):
	config = kda_config_factory(kda_implementation="chunkwise", kda_chunk_size=4, kda_gate_lower_bound=-8.0)

	assert config.kda_chunk_size == 4
