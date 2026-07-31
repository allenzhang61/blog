import torch

from model.LLMConfig import LLMConfig
from model.SimpleLLM import SimpleLLM
from model.RotaryEmbedding import RotaryEmbedding
from .deps import require_module
from .weights import load_into_model


def load_model(model_dir, device, dtype):
    """读取配置和权重，构造 SimpleLLM 并加载到指定设备。

    优化点：
    - 在 meta 设备上构造模型骨架，跳过随机初始化（原来会先随机填充全部权重再覆盖）。
    - 用 mmap 懒加载权重，逐张量直接映射到目标设备，避免多份内存拷贝。
    """
    require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"未找到 config.json：{config_path}")
    cfg = LLMConfig.from_json(config_path)

    # meta 设备：只建结构，不分配真实内存、不做随机初始化
    with torch.device("meta"):
        model = SimpleLLM(cfg)

    missing, shape_errors = load_into_model(model, model_dir, device, dtype)
    if missing or shape_errors:
        message = []
        if missing:
            message.append(f"缺少权重 key（前 20 个）：{missing[:20]}")
        if shape_errors:
            message.append(f"形状不匹配（前 10 个）：{shape_errors[:10]}")
        raise RuntimeError("\n".join(message))

    # RoPE 的 sin/cos cache 是 persistent=False 的 buffer，不在权重文件里，
    # meta 构造后仍是空张量，需要在目标设备上重新计算。
    _rebuild_rope_cache(model, cfg, device, dtype)

    model.eval()
    return model


def _rebuild_rope_cache(model, cfg, device, dtype):
    """在目标设备上重建 RotaryEmbedding 的非持久 sin/cos 缓存。"""
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            fresh = RotaryEmbedding(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
            )
            module.cos_cached = fresh.cos_cached.to(device=device)
            module.sin_cached = fresh.sin_cached.to(device=device)
