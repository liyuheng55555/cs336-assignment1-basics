import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2/(in_features + out_features))
        self.w: nn.Parameter = nn.Parameter.new(torch.rand(in_features, out_features))
        torch.nn.init.trunc_normal_(self.w, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("a b, ... b -> ... a", self.w.data, x)