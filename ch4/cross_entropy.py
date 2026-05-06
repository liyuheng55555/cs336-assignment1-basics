import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
        inputs: Float[Tensor, "... batch_size vocab_size"],
        targets: Int[Tensor, "... batch_size"],
) -> Float[Tensor, ""]:
    max_inputs = torch.max(inputs, dim=-1).values
    max_inputs = einops.rearrange(max_inputs, "... (batch vocab) -> ... batch vocab", vocab=1)
    inputs -= max_inputs
    exp_sum = einops.einsum(torch.exp(inputs), "... batch vocab -> ... batch")
    correct_logits = inputs[..., torch.arange(inputs.shape[-2]), targets] # [0,1...  targets[0],targets[1]...]
    loss = (torch.log(exp_sum) - correct_logits).sum() / targets.numel()
    return loss
