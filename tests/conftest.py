import pytest
import torch

from kimi_k3.model import KimiK3Config

_DEFAULT_CONFIG_KWARGS = dict(
	vocab_size=32,
	hidden_size=16,
	num_layers=1,
	rms_norm_eps=1e-6,
	initializer_range=0.02,
	max_seq_len=32,
	num_dense_layers=1,
	attn_res_block_size=1,
	layer_types=("kda",),
	kda_num_heads=2,
	kda_head_dim=4,
	kda_short_conv_kernel_size=3,
	kda_gate_lower_bound=-8.0,
	kda_full_rank_output_gate=True,
	kda_chunk_size=4,
	kda_implementation="chunkwise",
	mla_num_heads=2,
	mla_q_lora_rank=8,
	mla_kv_lora_rank=8,
	mla_qk_nope_head_dim=4,
	mla_qk_shared_head_dim=4,
	mla_v_head_dim=4,
	mla_output_gate=True,
	dense_intermediate_size=32,
	num_experts=2,
	num_experts_per_token=1,
	num_shared_experts=1,
	latent_moe_dim=8,
	expert_intermediate_size=16,
	latent_moe_use_norm=True,
	routed_scaling_factor=1.0,
	moe_renormalize=True,
	situ_beta=1.0,
	situ_linear_beta=1.0,
)


def make_kimi_k3_config(**overrides):
	"""Build a small KimiK3Config for tests, overriding only the fields a test cares about."""
	kwargs = {**_DEFAULT_CONFIG_KWARGS, **overrides}
	return KimiK3Config(**kwargs)


@pytest.fixture
def kda_config_factory():
	return make_kimi_k3_config


@pytest.fixture
def kda_config(kda_config_factory):
	return kda_config_factory()


@pytest.fixture(autouse=True)
def _seed_torch():
	torch.manual_seed(0)
