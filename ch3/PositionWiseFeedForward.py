import einops
import torch.nn


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_ff: int, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        self.d_ff = d_ff
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x: torch.Tensor):
        w1x = einops.einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = self._silu(w1x)
        w3x = einops.einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einops.einsum(self.w2, silu * w3x, "d_model d_ff, ... d_ff -> ... d_model")

    def _silu(self, x):
        return x * torch.sigmoid(x)