import einops
import torch.nn


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weights=None, device=None, dtype=None):
        self.d_model = d_model
        self.eps = eps
        if weights is not None:
            self.w = weights
        else:
            exit(-1) # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = self._rms(x).unsqueeze(-1)
        result = x / rms * self.w
        return result.to(in_dtype)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        sqaure = x * x + self.eps
        sum = einops.einsum(sqaure, "... d_model -> ...")
        return (sum / x.shape[-1]) ** 0.5
