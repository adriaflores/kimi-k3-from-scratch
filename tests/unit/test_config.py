import pytest


@pytest.mark.parametrize("num_experts_per_token", [0, 4, 5])
def test_num_experts_per_token_must_leave_room_for_balancer_threshold(kda_config_factory, num_experts_per_token):
	# The QuantileBalancer reads the (k+1)-th routed score, so top-k must be strictly below num_experts.
	with pytest.raises(ValueError, match="num_experts_per_token"):
		kda_config_factory(num_experts=4, num_experts_per_token=num_experts_per_token)


def test_largest_valid_num_experts_per_token_is_accepted(kda_config_factory):
	config = kda_config_factory(num_experts=4, num_experts_per_token=3)

	assert config.num_experts_per_token == 3
