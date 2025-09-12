
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.w: nn.Parameter = nn.Parameter(torch.rand(num_embeddings, embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]

