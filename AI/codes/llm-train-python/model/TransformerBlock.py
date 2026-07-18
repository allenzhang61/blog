import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttention
from .FeedForward import FeedForward
from .LayerNorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'],
        )
        self.feedforward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # 注意力模块中的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈网络中的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)

        # 将原始的输入加回到输出中
        x = x + shortcut
        return x
