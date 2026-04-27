import math

import einops
import torch

from ch3.Softmax import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    d_k: int = Q.shape[-1]
    qkt = einops.einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") # seq_len_q == seq_len_k == seq_len
    scaled_qkt = qkt / math.sqrt(d_k)
    if mask is not None:
        scaled_qkt = scaled_qkt.masked_fill(~mask, float("-inf"))
    softmax_qk = softmax(scaled_qkt, scaled_qkt.dim() - 1)
    result = einops.einsum(softmax_qk, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return result