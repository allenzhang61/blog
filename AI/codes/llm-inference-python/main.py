#!/usr/bin/env python3
"""A small self-implemented LLM forward CLI.

This example intentionally avoids Ollama, HTTP inference APIs, and
AutoModelForCausalLM. It uses Hugging Face only to download/cache files and to
load the tokenizer; the model forward path is implemented below.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


DEFAULT_PROMPT = "法国的首都是"
DEFAULT_MODEL = "deepseek-r1:8b"
DEFAULT_REPO_BY_MODEL = {
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用自实现 LLM forward 流程执行本地文本生成，不调用 Ollama API。",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=f"输入语句；默认：{DEFAULT_PROMPT}",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="输入语句；优先级高于位置参数。",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型标识；默认：{DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face repo id；不传时根据 --model 映射。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face revision，例如 main、commit hash 或 tag。",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face 缓存目录。",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="本地模型目录；提供后不从 Hugging Face 下载。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="最大生成 token 数；默认：32。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度；默认：0.7。",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="使用贪心解码，忽略 temperature。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="运行设备；默认自动选择。",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="权重 dtype；默认按设备自动选择。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载 tokenizer 时传给 transformers.AutoTokenizer。",
    )
    return parser.parse_args(argv)


def require_module(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"缺少依赖 {module_name}。请先执行：{install_hint}") from exc


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.input is not None:
        return args.input
    return DEFAULT_PROMPT


def resolve_repo_id(model: str, repo_id: Optional[str]) -> str:
    if repo_id:
        return repo_id
    if model in DEFAULT_REPO_BY_MODEL:
        return DEFAULT_REPO_BY_MODEL[model]
    return model


def resolve_model_dir(args: argparse.Namespace) -> Path:
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


def select_device(device_arg: str):
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


def select_dtype(dtype_arg: str, device):
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


@dataclass
class LLMConfig:
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
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: Path) -> "LLMConfig":
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
            num_key_value_heads=int(
                data.get("num_key_value_heads", data["num_attention_heads"])
            ),
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


def load_tokenizer(model_dir: Path, trust_remote_code: bool):
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


def find_weight_files(model_dir: Path) -> list[Path]:
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


def load_state_dict(model_dir: Path, device: str = "cpu") -> Dict[str, object]:
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    files = find_weight_files(model_dir)
    state: Dict[str, object] = {}
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


def repeat_kv(x, repeats: int):
    if repeats == 1:
        return x
    bsz, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, kv_heads, repeats, seq_len, head_dim)
    return x.reshape(bsz, kv_heads * repeats, seq_len, head_dim)


def build_model_classes():
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    nn = torch.nn
    F = torch.nn.functional

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x

    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, max_position: int, theta: float):
            super().__init__()
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            positions = torch.arange(max_position, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos(), persistent=False)
            self.register_buffer("sin_cached", emb.sin(), persistent=False)

        def forward(self, seq_len: int, device, dtype):
            cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
            sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
            return cos, sin

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(q, k, cos, sin):
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    class SelfAttention(nn.Module):
        def __init__(self, cfg: LLMConfig):
            super().__init__()
            self.num_heads = cfg.num_attention_heads
            self.num_kv_heads = cfg.num_key_value_heads
            self.head_dim = cfg.head_dim
            self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            self.rope = RotaryEmbedding(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
            )

        def forward(self, x):
            bsz, seq_len, hidden = x.shape
            q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rope(seq_len, x.device, x.dtype)
            q, k = apply_rope(q, k, cos, sin)
            repeats = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float("-inf"))
            attn = F.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.o_proj(out)

    class MLP(nn.Module):
        def __init__(self, cfg: LLMConfig):
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class DecoderLayer(nn.Module):
        def __init__(self, cfg: LLMConfig):
            super().__init__()
            self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.self_attn = SelfAttention(cfg)
            self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.mlp = MLP(cfg)

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x

    class SimpleLLM(nn.Module):
        def __init__(self, cfg: LLMConfig):
            super().__init__()
            self.cfg = cfg
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        def forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            hidden = self.norm(hidden)
            return self.lm_head(hidden)

    return SimpleLLM


def map_llama_keys(state: Dict[str, object]) -> Dict[str, object]:
    mapped: Dict[str, object] = {}
    prefix = "model."
    for key, value in state.items():
        target = key
        if target.startswith(prefix):
            target = target[len(prefix) :]
        if target == "embed_tokens.weight":
            target = "embed_tokens.weight"
        elif target == "norm.weight":
            target = "norm.weight"
        elif target.startswith("layers."):
            target = target
        elif target == "lm_head.weight":
            target = "lm_head.weight"
        mapped[target] = value
    if "lm_head.weight" not in mapped and "embed_tokens.weight" in mapped:
        mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
    return mapped


def load_model(model_dir: Path, device, dtype):
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"未找到 config.json：{config_path}")
    cfg = LLMConfig.from_json(config_path)
    SimpleLLM = build_model_classes()
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


def sample_next_token(logits, greedy: bool, temperature: float):
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    if greedy or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(model, tokenizer, prompt: str, args: argparse.Namespace, device):
    torch = require_module(
        "torch",
        "pip install -r AI/codes/llm-inference-python/requirements.txt",
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    generated = input_ids
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(generated)
            next_logits = logits[:, -1, :]
            next_id = sample_next_token(next_logits, args.greedy, args.temperature)
            generated = torch.cat([generated, next_id], dim=-1)
            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    prompt = resolve_prompt(args)
    try:
        model_dir = resolve_model_dir(args)
        log(f"模型文件目录已就绪：{model_dir}")
        log("开始加载模型到内存...")
        load_start = time.perf_counter()
        tokenizer = load_tokenizer(model_dir, args.trust_remote_code)
        device = select_device(args.device)
        dtype = select_dtype(args.dtype, device)
        model = load_model(model_dir, device, dtype)
        load_elapsed = time.perf_counter() - load_start
        log(f"模型加载完成，耗时 {load_elapsed:.2f}s，device={device}，dtype={dtype}")
        log("开始推理...")
        infer_start = time.perf_counter()
        text = generate(model, tokenizer, prompt, args, device)
        infer_elapsed = time.perf_counter() - infer_start
        log(f"推理完成，耗时 {infer_elapsed:.2f}s，max_new_tokens={args.max_new_tokens}")
        print(text)
        return 0
    except Exception as exc:
        print(f"推理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
