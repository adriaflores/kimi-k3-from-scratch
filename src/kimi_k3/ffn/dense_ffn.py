import torch
import torch.nn as nn

class SituGLU(nn.Module):

	def __init__(self, beta, linear_beta):
		super().__init__()

		self.beta = beta
		self.linear_beta = linear_beta

	def forward(self, x):
		gate, values = x.chunk(2, dim=-1) # (B, T, I), (B, T, I)
		gate, values = map(lambda x: x.to(torch.float), [gate, values])

		gate_branch = (self.beta * torch.tanh(gate/self.beta)) * torch.sigmoid(gate)
		value_branch = self.linear_beta * torch.tanh(values/self.linear_beta)

		return (gate_branch * value_branch).to(x.dtype)

class DenseFFN(nn.Module):

	def __init__(self, hidden_size, intermediate_size, situ: SituGLU):
		super().__init__()
		self.d_model = hidden_size
		self.intermediate_size = intermediate_size
		self.act_fn = situ

		self.gate = nn.Linear(self.d_model, self.intermediate_size, bias=False)
		self.up_proj = nn.Linear(self.d_model, self.intermediate_size, bias=False)
		self.down_proj = nn.Linear(self.intermediate_size, self.d_model,  bias=False)

	def forward(self, x):
		gate_up = torch.cat([self.gate(x), self.up_proj(x)], dim=-1) # (B, T, 2 * I)
		down_proj = self.down_proj(self.act_fn(gate_up)) # (B, T, C)

		return down_proj