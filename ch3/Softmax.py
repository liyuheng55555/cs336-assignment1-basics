import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_i = x.max(dim=i, keepdim=True)
    stabilized_x = x - max_i.values
    exp = torch.exp(stabilized_x)
    exp_sum = exp.sum(dim=i, keepdim=True)
    return exp / exp_sum