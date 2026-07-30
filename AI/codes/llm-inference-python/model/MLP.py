from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """SwiGLU 风格前馈网络，负责 token 表示的逐位置非线性变换。"""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        """执行 gate/up/down 三个投影组成的 SwiGLU MLP。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
