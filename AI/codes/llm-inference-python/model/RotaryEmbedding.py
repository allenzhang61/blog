from torch import nn
import torch


class RotaryEmbedding(nn.Module):
    """预计算 RoPE 的 sin/cos 表，用于给 Q/K 注入位置信息。"""

    def __init__(self, head_dim, max_position, theta):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device, dtype):
        """取出当前序列长度需要的 RoPE sin/cos，并移动到目标设备和 dtype。"""
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x):
    """将向量后半部分旋转到前半部分，用于 RoPE 复数旋转等价计算。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """把 RoPE 位置编码应用到 query 和 key。"""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
