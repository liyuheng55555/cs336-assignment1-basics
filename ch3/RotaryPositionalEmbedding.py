import math
import torch
from typing import Optional



class RotaryPositionalEmbedding(torch.nn.Module):


    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d = d_k
        self.max_seq_len = max_seq_len

        self.cos = torch.zeros(max_seq_len, self.d)
        self.sin = torch.zeros(max_seq_len, self.d)
        self.cal_cos_sin()



    def cal_cos_sin(self):
        for i in range(self.max_seq_len):
            for k in range(1, int(self.d / 2) + 2):
                theta_i_k = i / (self.theta ** ((2*(k - 1)) / self.d))
                self.cos[i][k] = math.cos(theta_i_k)
                self.sin[i][k] = math.sin(theta_i_k)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        x_flat = x.flatten()
        seq_len = x.shape[-2]
        result = torch.zeros(x.numel())
        base_index = 0
        while base_index < x.numel() / x.shape[-1]:
            position = token_positions.flatten()[base_index % token_positions.numel()]
            global_index = base_index * self.d
            for g in range(0, self.d, 2):
                k = int(g/2) + 1
                a = x_flat[global_index + g]
                b = x_flat[global_index + g + 1]
                result[global_index + g] = a * self.cos[position][k] - b * self.sin[position][k]
                result[global_index + g + 1] = a * self.sin[position][k] + b * self.cos[position][k]
            base_index += 1

        result_reshape = result.reshape(x.shape)
        return result_reshape


