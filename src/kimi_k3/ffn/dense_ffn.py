import torch
import torch.nn as nn


def _situ_glu_impl(x, beta, linear_beta):
	"""SiTU-GLU in FP32 internally, preserving the input dtype at output."""

	input_dtype = x.dtype

	gate, values = x.float().chunk(2, dim=-1)

	gate_branch = (beta * torch.tanh(gate / beta) * torch.sigmoid(gate))
	value_branch = (linear_beta * torch.tanh(values / linear_beta))

	return (gate_branch * value_branch).to(input_dtype)


# One shared compiled callable for all SiTU instances. Compilation itself remains lazy until the first CUDA invocation.
_compiled_situ_glu = torch.compile(_situ_glu_impl, backend="inductor")


class SituGLU(nn.Module):

	def __init__(self, beta, linear_beta):
		super().__init__()

		self.beta = beta
		self.linear_beta = linear_beta

	def forward(self, x):
		return _situ_glu_impl(x, self.beta, self.linear_beta)

	def compiled(self, x):
		return _compiled_situ_glu(x, self.beta, self.linear_beta)

class DenseFFN(nn.Module):

	def __init__(self, hidden_size, intermediate_size, situ: SituGLU, *, compile_situ=False):
		super().__init__()

		self.d_model = hidden_size
		self.intermediate_size = intermediate_size
		self.act_fn = situ
		self.compile_situ = compile_situ

		self.gate_up_proj = nn.Linear(self.d_model, 2 * self.intermediate_size, bias=False)

		self.down_proj = nn.Linear(self.intermediate_size, self.d_model, bias=False)

	def forward(self, x):
		gate_up = self.gate_up_proj(x)

		if (
			self.compile_situ
			and gate_up.is_cuda
			and self.training
			and torch.is_grad_enabled()
		):
			torch._dynamo.maybe_mark_dynamic(gate_up, 0)
			activated = self.act_fn.compiled(gate_up)
		else:
			activated = self.act_fn(gate_up)

		return self.down_proj(activated)