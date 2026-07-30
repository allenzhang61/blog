from .deps import require_module


def select_device(device_arg):
    """根据参数选择 PyTorch 运行设备，auto 时优先 CUDA、再 MPS、最后 CPU。"""
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但 torch.cuda.is_available() 为 False。")
    if device_arg == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("指定了 mps，但当前 PyTorch MPS 不可用。")
    return torch.device(device_arg)


def select_dtype(dtype_arg, device):
    """根据参数和设备选择权重 dtype。"""
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32
