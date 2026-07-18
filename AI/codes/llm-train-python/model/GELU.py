import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
