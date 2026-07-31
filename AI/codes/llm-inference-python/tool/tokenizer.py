from .deps import require_module


def load_tokenizer(model_dir, trust_remote_code):
    """从本地模型目录加载 tokenizer，用于文本与 token id 的转换。

    优先使用 PreTrainedTokenizerFast 直接基于 tokenizer.json 加载：
    某些模型（如 DeepSeek-R1-Distill-Llama）的 tokenizer_config 里写的是
    LlamaTokenizerFast，但其 tokenizer.json 实际是 GPT-2 byte-level BPE，
    用 AutoTokenizer 会误加载成 sentencepiece 风格的 Llama tokenizer，导致
    解码出现裸露的 Ġ/Ċ、空格丢失、中文被编码为空等问题。
    PreTrainedTokenizerFast.from_pretrained 会正确应用 byte-level decoder，
    同时读取 tokenizer_config.json 里的 bos/eos 等特殊 token。
    """
    transformers = require_module(
        "transformers",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            return transformers.PreTrainedTokenizerFast.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )
        except Exception:
            pass
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
