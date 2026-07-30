from torch import nn

from .DecoderLayer import DecoderLayer
from .RMSNorm import RMSNorm


class SimpleLLM(nn.Module):
    """简洁 Llama-like causal LM：embedding、decoder blocks、norm、lm head。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        """从 token ids 计算每个位置的词表 logits。"""
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)
