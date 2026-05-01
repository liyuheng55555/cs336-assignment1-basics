import math

import torch.optim
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float,
            weight_decay: float,
            betas: tuple[float, float],
            eps: float,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                assert isinstance(p, torch.Tensor)
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m: torch.Tensor = state.get("m", torch.zeros(p.shape))
                v: torch.Tensor = state.get("v", torch.zeros(p.shape))
                g = p.grad.data

                p.data -= lr * weight_decay * p.data

                lr_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)

                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * (g ** 2)

                p.data -= lr_t * m / (v.sqrt() + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v