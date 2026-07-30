from pathlib import Path

from .deps import require_module


DEFAULT_REPO_BY_MODEL = {
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


def resolve_repo_id(model, repo_id):
    """根据命令行 repo-id 或模型别名确定 Hugging Face repo id。"""
    if repo_id:
        return repo_id
    if model in DEFAULT_REPO_BY_MODEL:
        return DEFAULT_REPO_BY_MODEL[model]
    return model


def resolve_model_dir(args):
    """解析模型目录；优先使用本地目录，否则从 Hugging Face 下载并返回缓存路径。"""
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        if not model_dir.exists():
            raise RuntimeError(f"本地模型目录不存在：{model_dir}")
        return model_dir

    hf_hub = require_module(
        "huggingface_hub",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    repo_id = resolve_repo_id(args.model, args.repo_id)
    allow_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "pytorch_model*.bin.index.json",
    ]
    try:
        downloaded = hf_hub.snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            allow_patterns=allow_patterns,
            local_files_only=False,
        )
        return Path(downloaded)
    except Exception as exc:
        raise RuntimeError(
            "从 Hugging Face 下载模型文件失败。请检查网络、repo id、revision、"
            "鉴权 token，或使用 --model-dir 指向已有本地模型目录。\n"
            f"repo_id={repo_id}\n原始错误：{exc}"
        ) from exc
