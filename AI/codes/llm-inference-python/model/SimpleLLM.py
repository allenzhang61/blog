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

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        """从 token ids 计算每个位置的词表 logits。

        past_key_values：每层的 (k, v) 缓存列表，None 表示无历史。
        use_cache：为 True 时返回 (logits, new_past_key_values) 以支持增量解码；
        为 False 时仅返回 logits，保持与原有调用方式兼容。
        """
        position_offset = 0
        if past_key_values is not None and past_key_values[0] is not None:
            position_offset = past_key_values[0][0].shape[2]
        hidden = self.embed_tokens(input_ids)
        new_past = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden, new_kv = layer(
                hidden, past_kv=layer_past, position_offset=position_offset
            )
            new_past.append(new_kv)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        if use_cache:
            return logits, new_past
        return logits
