import torch.nn as nn
import torch
import math
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # batch_size, sequence_length, embedding_dimensionality(n_embd)
        # 批大小、序列长度、嵌入维度
        B, T, C = x.size()

        # x = (B,T,n_embd)
        # c_attn(x) = (B,T, 3*n_embd)
        # c_attn(x).split(self.b_embd, dim=2) = (B,T,n_embd)
        q, k, v = self.c_attn(x).split(self.b_embd, dim=2)
        # (B,n_head,T,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B,n_head,T,hs) * (B,n_head,hs,T) = (B,n_head,T,T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 找到为 0 的位置设置为-inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # (B,n_head,T,T) * (B,n_head,T,hs) = (B,n_head,T,hs)
        y = att @ v
        # y.transpose(1, 2) = (B,T,n_head,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
