import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """保存 decoder-only LLM forward 所需的核心结构参数。"""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    @property
    def head_dim(self):
        """返回单个 attention head 的维度。"""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path):
        """从 Hugging Face config.json 读取并校验模型结构配置。"""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        required = [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(f"config.json 缺少必要字段：{missing}")
        cfg = cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
            max_position_embeddings=int(data.get("max_position_embeddings", 4096)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
        )
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise RuntimeError(
                "hidden_size 必须能被 num_attention_heads 整除："
                f"{cfg.hidden_size} vs {cfg.num_attention_heads}"
            )
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
            raise RuntimeError(
                "num_attention_heads 必须能被 num_key_value_heads 整除："
                f"{cfg.num_attention_heads} vs {cfg.num_key_value_heads}"
            )
        return cfg
