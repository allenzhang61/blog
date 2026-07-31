#!/usr/bin/env python3
"""自实现 LLM forward 推理命令行入口。"""

import argparse
import sys
import time

from generate.generate import generate
from tool.model_files import resolve_model_dir
from tool.model_loader import load_model
from tool.runtime import select_device, select_dtype
from tool.tokenizer import load_tokenizer


DEFAULT_PROMPT = "法国的首都是"
DEFAULT_MODEL = "deepseek-r1:8b"


def parse_args(argv=None):
    """解析命令行参数，定义 prompt、模型来源、生成参数和运行设备。"""
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
    parser.add_argument("--repo-id", default=None, help="Hugging Face repo id；不传时根据 --model 映射。")
    parser.add_argument("--revision", default=None, help="Hugging Face revision，例如 main、commit hash 或 tag。")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face 缓存目录。")
    parser.add_argument("--model-dir", default=None, help="本地模型目录；提供后不从 Hugging Face 下载。")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="最大生成 token 数；默认：32。")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度；默认：0.7。")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码，忽略 temperature。")
    parser.add_argument(
        "--no-kv-cache",
        dest="use_kv_cache",
        action="store_false",
        help="关闭 KV Cache，每步重新前向整个序列（用于对比基线）。",
    )
    parser.set_defaults(use_kv_cache=True)
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
    parser.add_argument("--trust-remote-code", action="store_true", help="加载 tokenizer 时传给 transformers.AutoTokenizer。")
    return parser.parse_args(argv)


def resolve_prompt(args):
    """按优先级选择输入语句：--prompt > 位置参数 > 默认 prompt。"""
    if args.prompt is not None:
        return args.prompt
    if args.input is not None:
        return args.input
    return DEFAULT_PROMPT


def log(message):
    """把运行日志输出到 stderr，避免污染 stdout 中的模型生成文本。"""
    print(message, file=sys.stderr, flush=True)


def main(argv=None):
    """串联模型文件准备、模型加载、文本生成和错误处理。"""
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
