#!/usr/bin/env python3
"""Qwen3.5-4B 本地 CUDA 推理入口。"""

import argparse
import sys
import time


MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_PROMPT = "介绍一下 TCP 三次握手"


def parse_args(argv=None):
    """解析 Qwen3.5-4B 推理参数。"""
    parser = argparse.ArgumentParser(
        description="在本地加载 Qwen/Qwen3.5-4B 并执行文本生成。",
    )
    parser.add_argument("input", nargs="?", default=None, help=f"输入语句；默认：{DEFAULT_PROMPT}")
    parser.add_argument("-p", "--prompt", default=None, help="输入语句；优先级高于位置参数。")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face 缓存目录。")
    parser.add_argument("--revision", default=None, help="Hugging Face revision，例如 main、commit hash 或 tag。")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成 token 数；默认：128。")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度；默认：0.7。")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码，忽略 temperature。")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="运行设备；默认：cuda。",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="模型 dtype；RTX 3080 默认建议 float16。",
    )
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


def resolve_torch_dtype(torch, dtype_name):
    """把 CLI dtype 名称转换为 torch dtype。"""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_inputs(processor, prompt):
    """按 Qwen chat template 构造纯文本输入。"""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def main(argv=None):
    """加载 Qwen/Qwen3.5-4B 并生成文本。"""
    args = parse_args(argv)
    prompt = resolve_prompt(args)
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False。")

        dtype = resolve_torch_dtype(torch, args.dtype)
        device_map = {"": 0} if args.device == "cuda" else {"": "cpu"}

        log(f"开始加载 {MODEL_ID} ...")
        load_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            revision=args.revision,
            cache_dir=args.cache_dir,
            dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        model.eval()
        load_elapsed = time.perf_counter() - load_start
        log(f"模型加载完成，耗时 {load_elapsed:.2f}s，device={args.device}，dtype={dtype}")

        inputs = build_inputs(processor, prompt).to(model.device)
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": not args.greedy,
        }
        if not args.greedy:
            generate_kwargs["temperature"] = args.temperature

        log("开始推理...")
        infer_start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)
        infer_elapsed = time.perf_counter() - infer_start
        log(f"推理完成，耗时 {infer_elapsed:.2f}s，max_new_tokens={args.max_new_tokens}")

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated[:, input_len:]
        print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
        return 0
    except Exception as exc:
        print(f"推理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
