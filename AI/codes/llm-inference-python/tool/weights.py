from .deps import require_module


def find_weight_files(model_dir):
    """在模型目录中查找 safetensors 或 PyTorch bin 权重文件。"""
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files
    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if bin_files:
        return bin_files
    raise RuntimeError(
        f"未找到权重文件：{model_dir}\n"
        "需要 *.safetensors 或 pytorch_model*.bin。"
    )


def load_state_dict(model_dir, device="cpu"):
    """读取一个或多个权重分片并合并成 state dict。"""
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    files = find_weight_files(model_dir)
    state = {}
    for path in files:
        if path.suffix == ".safetensors":
            try:
                safetensors_torch = require_module(
                    "safetensors.torch",
                    "pip install -r AI/codes/llm-inference-python/requirements.txt",
                )
                shard = safetensors_torch.load_file(str(path), device=device)
            except Exception as exc:
                raise RuntimeError(f"读取 safetensors 失败：{path}\n{exc}") from exc
        else:
            shard = torch.load(path, map_location=device)
        state.update(shard)
    return state


def map_llama_keys(state):
    """把 Hugging Face Llama-like 权重 key 映射到本示例模型的 key。"""
    mapped = {}
    prefix = "model."
    for key, value in state.items():
        target = key
        if target.startswith(prefix):
            target = target[len(prefix) :]
        mapped[target] = value
    if "lm_head.weight" not in mapped and "embed_tokens.weight" in mapped:
        mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
    return mapped
