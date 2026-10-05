import torch.nn as nn
import torch.nn.functional as F

class MixedPrecisionRMSNorm(nn.RMSNorm):
	def forward(self, x):
		return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype) if self.weight is not None else None, self.eps)