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

    def forward(self, x, past_kv=None, position_offset=0):
        """计算 masked self-attention，并输出投影回 hidden size。

        past_kv：(past_k, past_v) 历史缓存，None 表示 prefill（首次全量前向）。
        position_offset：当前输入在完整序列中的起始位置，用于 RoPE 取正确的 sin/cos。
        返回 (输出, (new_k, new_v))，new_k/new_v 是拼接历史后的完整缓存。
        """
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE：按当前 token 在完整序列中的绝对位置取 sin/cos
        cos, sin = self.rope(position_offset + seq_len, x.device, x.dtype)
        cos = cos[position_offset:]
        sin = sin[position_offset:]
        q, k = apply_rope(q, k, cos, sin)
        # 拼接历史 KV
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)
        repeats = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        kv_len = k.shape[2]
        # causal mask：query 的绝对位置为 position_offset+i，可见 key 位置 <= 自身。
        # 增量解码（seq_len==1）时新 token 可见全部历史，无需 mask。
        if seq_len > 1:
            mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=x.device),
                diagonal=1 + (kv_len - seq_len),
            )
            attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float("-inf"))
        attn = F.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o_proj(out), new_kv
