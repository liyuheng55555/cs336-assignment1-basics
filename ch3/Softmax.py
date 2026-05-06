import torch


def softmax(x: torch.Tensor, i: int, temp: float = 1) -> torch.Tensor:
    max_i = x.max(dim=i, keepdim=True)
    stabilized_x = x - max_i.values
    temp_x = x / temp
    exp = torch.exp(temp_x)
    exp_sum = exp.sum(dim=i, keepdim=True)
    return exp / exp_sum