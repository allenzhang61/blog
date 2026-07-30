from .deps import require_module


def load_tokenizer(model_dir, trust_remote_code):
    """从本地模型目录加载 tokenizer，用于文本与 token id 的转换。"""
    transformers = require_module(
        "transformers",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    try:
        return transformers.AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"加载 tokenizer 失败：{model_dir}\n"
            "请确认目录中包含 tokenizer.json/tokenizer.model 等文件。\n"
            f"原始错误：{exc}"
        ) from exc
