from torch import nn
import torch


class RMSNorm(nn.Module):
    """RMSNorm 归一化层，Llama-like 模型中常用在 attention/MLP 前。"""

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        """对最后一维做 RMS 归一化并乘以可学习缩放参数。"""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
