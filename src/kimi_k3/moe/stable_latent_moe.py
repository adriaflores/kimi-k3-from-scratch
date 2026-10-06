import torch
import torch.nn.functional as F
from torch import nn

from src.kimi_k3.ffn import DenseFFN as Expert
from src.kimi_k3.ffn import SituGLU

from src.kimi_k3.utils.rmsn import MixedPrecisionRMSNorm

class QuantileBalancer(nn.Module):

	def __init__(self, num_experts, k):
		super().__init__()

		self.num_experts = num_experts
		self.top_k = k

		self.register_buffer("biases", torch.zeros(self.num_experts), persistent=True)

	def get_biases(self):
		return self.biases

	@torch.no_grad
	def update(self, top_scores, router_scores):
		p = self.top_k / self.num_experts

		thresh = top_scores.view(-1, self.top_k + 1)[:, -1].unsqueeze(-1) # (BT, 1)
		relative_thresh = thresh - router_scores # (BT, E)
		new_biases = torch.quantile(relative_thresh, p, dim=0, interpolation="higher") # (E)
		new_biases = new_biases - new_biases.mean()

		self.biases.copy_(new_biases)


class StableLatentMoE(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.d_model = config.hidden_size
		self.num_routed_experts = config.num_experts
		self.latent_moe_dim = config.latent_moe_dim
		self.num_shared_experts = config.num_shared_experts
		self.expert_intermediate_size = config.expert_intermediate_size
		self.top_k = config.num_experts_per_token
		self.latent_moe_use_norm = config.latent_moe_use_norm
		self.moe_renormalize = config.moe_renormalize
		self.routed_scaling_factor = config.routed_scaling_factor

		self.situ = SituGLU(beta=config.situ_beta, linear_beta=config.situ_linear_beta)

		self.shared_experts = Expert(hidden_size=self.d_model, intermediate_size=self.num_shared_experts * self.expert_intermediate_size, situ=self.situ)

		self.router = nn.Linear(self.d_model, self.num_routed_experts, bias=False)
		self.down_latent_proj = nn.Linear(self.d_model, self.latent_moe_dim, bias=False)
		self.experts = nn.ModuleList(
			[
				Expert(
					hidden_size=self.latent_moe_dim,
					intermediate_size=self.expert_intermediate_size,
					situ=self.situ,
					compile_situ=True
				)
				for _ in range(self.num_routed_experts)

			]
		)
		self.post_router_rmsnorms = MixedPrecisionRMSNorm(self.latent_moe_dim, eps=config.rms_norm_eps)
		self.up_latent_proj = nn.Linear(self.latent_moe_dim, self.d_model, bias=False)

		self.quantile_balancer = QuantileBalancer(num_experts=self.num_routed_experts, k=self.top_k)
		# Optional Python counters owned by a diagnostics context, not state_dict.
		self._expert_load_counts = None


	def forward(self, x):
		B, T, _ = x.shape

		# Shared part
		shared = self.shared_experts(x) # (B, T, C)

		#Routed part
		# Explicit .float() inputs alone do not stop autocast from lowering F.linear.
		with torch.autocast(device_type=x.device.type, enabled=False):
			router_logits = F.linear(x.to(torch.float32), self.router.weight.to(torch.float32), bias=None) # (B, T, E)
			router_scores = torch.sigmoid(router_logits)
			biases = self.quantile_balancer.get_biases() # (E)
			corrected_scores = router_scores + biases # (B, T, E)
			top_scores, top_indices = torch.topk(corrected_scores, k=self.top_k + 1, dim=-1) # (B, T, K + 1), (B, T, K + 1)
			router_scores = router_scores.reshape(B * T, -1) # (BT, E)
			if self.training:
				self.quantile_balancer.update(top_scores=top_scores, router_scores=router_scores)
		top_indices = top_indices[..., :-1] # (B, T, K)
		top_indices = top_indices.reshape(B * T, -1) # (BT, K)

		x_flat = x.reshape(B * T, -1) # (BT, C)
		z = self.down_latent_proj(x_flat) # (BT, D_latent)

		weights = torch.gather(router_scores, dim=-1, index=top_indices) # (BT, K)
		if self.moe_renormalize:
			weights = F.normalize(weights, p=1, dim=-1)
		
		expert_ids = top_indices.flatten() # (BTK)
		flat_weights = weights.flatten() # (BTK)

		order = torch.argsort(expert_ids, stable=True)

		sorted_token_ids = order // self.top_k
		sorted_weights = flat_weights[order]

		counts = torch.bincount(expert_ids, minlength=self.num_routed_experts)
		ends = torch.cumsum(counts, dim=0).tolist()

		if self.training and self._expert_load_counts is not None:
			start = 0
			for expert_j, end in enumerate(ends):
				self._expert_load_counts[expert_j] += end - start
				start = end

		routed_output = self._forward_grouped_experts(z, sorted_token_ids, sorted_weights, ends)

		routed_output = routed_output.view(B, T, self.latent_moe_dim) # (B, T, D_latent)
		if self.latent_moe_use_norm:
			routed_output = self.post_router_rmsnorms(routed_output)
		routed = self.up_latent_proj(routed_output) # (B, T, C)

		output = shared + routed

		return output


	def _forward_naive_experts(self, z, sorted_token_ids, sorted_weights, ends):
		# Readable reference for the paper's per-expert computation; forward uses grouped execution.
		routed_output = z.new_zeros(z.shape)
		start = 0
		for expert_j in range(self.num_routed_experts):
			end = ends[expert_j]
			token_indices = sorted_token_ids[start:end] # (q_j)
			expert_weights = sorted_weights[start:end].unsqueeze(-1) * self.routed_scaling_factor # (q_j, 1)
			start = end

			if token_indices.numel() == 0:
				continue

			z_e = z[token_indices] # (q_j, D_latent), where q_j is the number of tokens assigned to the j-th expert
			y_e = self.experts[expert_j](z_e) # (q_j, D_latent)
			expert_weights = expert_weights.to(y_e.dtype)
			expert_output = y_e * expert_weights # (q_j, D_latent)

			routed_output.index_add_(0, token_indices, expert_output)

		return routed_output


	def _forward_grouped_experts(self, z, sorted_token_ids, sorted_weights, ends):
		# Empty experts stay outside the graph, preserving grad=None and sparse AdamW updates.
		active_ids = [j for j, end in enumerate(ends) if end > (ends[j - 1] if j else 0)]
		offsets = torch.tensor([ends[j] for j in active_ids], device=z.device, dtype=torch.int32)
		z_sorted = z[sorted_token_ids] # (BTK, D_latent)

		# grouped_mm does not autocast its operands; only the temporary stacks change dtype.
		gate_up_weights = torch.stack([self.experts[j].gate_up_proj.weight for j in active_ids]).to(z.dtype)
		gate_up = F.grouped_mm(z_sorted, gate_up_weights.transpose(-2, -1), offs=offsets)
		if gate_up.is_cuda and self.training and torch.is_grad_enabled():
			torch._dynamo.maybe_mark_dynamic(gate_up, 0)
			activated = self.situ.compiled(gate_up)
		else:
			activated = self.situ(gate_up)

		down_weights = torch.stack([self.experts[j].down_proj.weight for j in active_ids]).to(activated.dtype)
		y_sorted = F.grouped_mm(activated, down_weights.transpose(-2, -1), offs=offsets)
		expert_weights = (sorted_weights.unsqueeze(-1) * self.routed_scaling_factor).to(y_sorted.dtype)

		routed_output = z.new_zeros(z.shape)
		routed_output.index_add_(0, sorted_token_ids, y_sorted * expert_weights)
		return routed_output
