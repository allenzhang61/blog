from model.LLMConfig import LLMConfig
from model.SimpleLLM import SimpleLLM
from .deps import require_module
from .weights import load_state_dict, map_llama_keys


def load_model(model_dir, device, dtype):
    """读取配置和权重，构造 SimpleLLM 并加载到指定设备。"""
    require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"未找到 config.json：{config_path}")
    cfg = LLMConfig.from_json(config_path)
    model = SimpleLLM(cfg)
    raw_state = load_state_dict(model_dir, device="cpu")
    mapped_state = map_llama_keys(raw_state)
    expected = model.state_dict()
    missing = [key for key in expected.keys() if key not in mapped_state]
    shape_errors = []
    for key, tensor in mapped_state.items():
        if key in expected and tuple(tensor.shape) != tuple(expected[key].shape):
            shape_errors.append((key, tuple(tensor.shape), tuple(expected[key].shape)))
    if missing or shape_errors:
        message = []
        if missing:
            message.append(f"缺少权重 key（前 20 个）：{missing[:20]}")
        if shape_errors:
            message.append(f"形状不匹配（前 10 个）：{shape_errors[:10]}")
        raise RuntimeError("\n".join(message))
    model.load_state_dict(mapped_state, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
