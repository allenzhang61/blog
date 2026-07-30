import math

import torch
from torch import nn
import torch.nn.functional as F

from .RotaryEmbedding import RotaryEmbedding, apply_rope


def repeat_kv(x, repeats):
    """在 GQA/MQA 中把 key/value head 重复到 query head 数量。"""
    if repeats == 1:
        return x
    bsz, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, kv_heads, repeats, seq_len, head_dim)
    return x.reshape(bsz, kv_heads * repeats, seq_len, head_dim)


class SelfAttention(nn.Module):
    """带 RoPE 和 causal mask 的 Llama-like 自注意力层。"""

    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
        )

    def forward(self, x):
        """计算 masked self-attention，并输出投影回 hidden size。"""
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        repeats = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float("-inf"))
        attn = F.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o_proj(out)
