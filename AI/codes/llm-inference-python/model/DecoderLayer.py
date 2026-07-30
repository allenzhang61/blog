from torch import nn

from .MLP import MLP
from .RMSNorm import RMSNorm
from .SelfAttention import SelfAttention


class DecoderLayer(nn.Module):
    """单个 decoder-only Transformer block：norm、attention、MLP 和残差。"""

    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = SelfAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x):
        """依次执行 attention 子层和 MLP 子层，并保留残差连接。"""
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
