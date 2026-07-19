import torch.nn as nn
from Block import Block
import torch
import math
from torch.nn import functional as F


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 准备参数
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.embd_pdrop = config.embd_pdrop
        preset = {
            # names follow the huggingface naming conventions；名称遵循 HuggingFace 约定
            # GPT-1；GPT-1 配置
            'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
            # GPT-2 configs；GPT-2 系列配置
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
            # Gophers；Gopher 风格的小配置
            'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512),
            # (there are a number more...)；还有更多可选配置未列出
            # I made these tiny models up；下面这些迷你模型是本项目自定义的
            'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
            'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
            'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48),
        }
        self.n_layer = preset[config.model_type].n_layer
        self.n_head = preset[config.model_type].n_head
        self.n_embd = preset[config.model_type].n_embd

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.n_embd),
            wpe=nn.Embedding(self.block_size, self.n_embd),
            drop=nn.Dropout(self.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.ln_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
