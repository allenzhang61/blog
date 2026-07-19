import torch
import torch.nn as nn
import math
from CausalSelfAttention import CausalSelfAttention


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj = nn.Linear(4 *config.n_embd,  config.n_embd),
            act = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x : m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlpf(self.ln_2(x))
        return x


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    GELU 激活函数实现，与 Google BERT 仓库和 OpenAI GPT 中使用的形式一致。
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
