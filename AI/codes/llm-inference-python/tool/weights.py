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


def map_llama_key(key):
    """把单个 Hugging Face Llama-like 权重 key 映射到本示例模型的 key。"""
    prefix = "model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def load_into_model(model, model_dir, device, dtype):
    """用 mmap 懒加载权重，逐张量直接赋值到模型参数，避免多份内存拷贝。

    要求 model 已在 meta 设备上构造（不分配真实内存、不随机初始化）。
    返回 (missing, shape_errors)，供调用方做校验。
    """
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    safe_open = require_module(
        "safetensors",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    ).safe_open

    expected = dict(model.state_dict())
    device_str = str(device)
    seen = set()
    shape_errors = []
    embed_tensor = None

    files = find_weight_files(model_dir)
    use_safetensors = files[0].suffix == ".safetensors"

    def assign(target_key, tensor):
        if target_key not in expected:
            return
        if tuple(tensor.shape) != tuple(expected[target_key].shape):
            shape_errors.append(
                (target_key, tuple(tensor.shape), tuple(expected[target_key].shape))
            )
            return
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        _set_module_tensor(model, target_key, tensor)
        seen.add(target_key)

    for path in files:
        if use_safetensors:
            # framework="pt" + device：safetensors 走 mmap，直接映射到目标设备
            with safe_open(str(path), framework="pt", device=device_str) as handle:
                for hf_key in handle.keys():
                    target = map_llama_key(hf_key)
                    tensor = handle.get_tensor(hf_key)
                    if target == "embed_tokens.weight":
                        embed_tensor = tensor
                    assign(target, tensor)
        else:
            shard = torch.load(path, map_location=device_str, mmap=True)
            for hf_key, tensor in shard.items():
                target = map_llama_key(hf_key)
                if target == "embed_tokens.weight":
                    embed_tensor = tensor
                assign(target, tensor)

    # tie weights：lm_head 复用 embedding 权重
    if "lm_head.weight" not in seen and embed_tensor is not None:
        assign("lm_head.weight", embed_tensor)

    missing = [key for key in expected.keys() if key not in seen]
    return missing, shape_errors


def _set_module_tensor(model, key, tensor):
    """把 tensor 设为 model 中 key 对应的 parameter/buffer，替换 meta 占位。"""
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    module = model
    *path, attr = key.split(".")
    for name in path:
        module = getattr(module, name)
    current = getattr(module, attr)
    if isinstance(current, torch.nn.Parameter):
        module.__setattr__(attr, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        module.__setattr__(attr, tensor)
