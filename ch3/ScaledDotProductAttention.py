import math

import einops
import torch

from ch3.Softmax import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    d_k: int = Q.shape[-1]
    qkt = einops.einsum(Q, K, "... query_len d_k, ... key_len d_k -> ... query_len key_len")
    scaled_qkt = qkt / math.sqrt(d_k)
    if mask is not None:
        scaled_qkt = scaled_qkt.masked_fill(~mask, float("-inf"))
    softmax_qk = softmax(scaled_qkt, scaled_qkt.dim() - 1)
    result = einops.einsum(softmax_qk, V, "... query_len key_len, ... key_len d_v -> ... query_len d_v")
    return result