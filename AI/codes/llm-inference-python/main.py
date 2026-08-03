#!/usr/bin/env python3
"""Qwen3.5-4B 本地 CUDA 推理入口。"""

import argparse
import contextlib
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        "--disable-thinking",
        action="store_true",
        help="调用 Qwen chat template 的 enable_thinking=False，生成空 think 段后直接回答。",
    )
    parser.add_argument(
        "--fast-decode",
        action="store_true",
        help="使用手写增量 decode loop，绕过 Transformers 通用 generate。当前使用动态 KV cache。",
    )
    parser.add_argument(
        "--static-cache",
        action="store_true",
        help="使用 static KV cache；适合常驻进程预热后提速，单次 CLI 可能包含编译开销。",
    )
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
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="使用手写 generate loop 输出分阶段耗时 JSON。",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="正式计时前先跑几次同配置生成，但不纳入统计；用于排除冷启动影响。默认：0。",
    )
    parser.add_argument(
        "--torch-profiler",
        default=None,
        metavar="DIR",
        help="启用 PyTorch profiler，并把 TensorBoard trace 写入指定目录。",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="添加 NVTX range，配合 nsys profile 查看 CPU/CUDA 时间线。",
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


def resolve_torch_dtype(dtype_name):
    """把 CLI dtype 名称转换为 torch dtype。"""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_inputs(tokenizer, prompt, disable_thinking):
    """按 Qwen chat template 构造纯文本输入。"""
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=not disable_thinking,
    )
    return tokenizer(text, return_tensors="pt")


def cuda_sync():
    """同步 CUDA，保证计时包含 GPU 异步执行时间。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def nvtx_range(enabled, name):
    """按需添加 NVTX range，供 Nsight Systems 展示阶段边界。"""
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def sample_next_token(logits, args):
    """从最后一步 logits 采样下一个 token。"""
    if args.greedy or args.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / args.temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def summarize_seconds(values):
    """汇总秒级列表，保留原始样本，便于后续分析抖动。"""
    if not values:
        return {
            "total_s": 0.0,
            "avg_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "samples_s": [],
        }
    total = sum(values)
    return {
        "total_s": round(total, 6),
        "avg_s": round(total / len(values), 6),
        "min_s": round(min(values), 6),
        "max_s": round(max(values), 6),
        "samples_s": [round(value, 6) for value in values],
    }


def profiled_generate(model, inputs, args):
    """手写增量解码，拆分 prefill、decode、采样和 CPU gap。"""
    if args.static_cache:
        raise RuntimeError("--profile-timing 暂不支持 --static-cache，请使用默认动态 KV cache。")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_id = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = {int(item) for item in eos_id}
    elif eos_id is None:
        eos_ids = set()
    else:
        eos_ids = {int(eos_id)}

    generated_chunks = [input_ids]
    token_records = []
    sample_times = []
    decode_model_times = []
    cpu_gap_times = []

    with nvtx_range(args.nvtx, "prefill"):
        cuda_sync()
        start = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        cuda_sync()
        prefill_s = time.perf_counter() - start

    past = outputs.past_key_values

    with nvtx_range(args.nvtx, "sample_from_prefill"):
        cuda_sync()
        start = time.perf_counter()
        next_id = sample_next_token(outputs.logits[:, -1, :], args)
        cuda_sync()
        sample_s = time.perf_counter() - start
    sample_times.append(sample_s)
    generated_chunks.append(next_id)
    token_records.append(
        {
            "token_index": 0,
            "source": "prefill_logits",
            "decode_model_s": 0.0,
            "sample_s": round(sample_s, 6),
            "cpu_gap_before_model_s": 0.0,
            "token_id": int(next_id.item()),
        }
    )

    last_gpu_done = time.perf_counter()
    if eos_ids and int(next_id.item()) in eos_ids:
        generated = torch.cat(generated_chunks, dim=-1)
        return generated, build_timing_report(
            input_ids.shape[-1],
            len(token_records),
            prefill_s,
            decode_model_times,
            sample_times,
            cpu_gap_times,
            token_records,
        )

    for token_index in range(1, args.max_new_tokens):
        before_model = time.perf_counter()
        cpu_gap_s = before_model - last_gpu_done
        cpu_gap_times.append(cpu_gap_s)
        with nvtx_range(args.nvtx, f"decode_model_token_{token_index}"):
            cuda_sync()
            start = time.perf_counter()
            outputs = model(input_ids=next_id, past_key_values=past, use_cache=True)
            cuda_sync()
            decode_model_s = time.perf_counter() - start
        past = outputs.past_key_values
        decode_model_times.append(decode_model_s)

        with nvtx_range(args.nvtx, f"sample_token_{token_index}"):
            cuda_sync()
            start = time.perf_counter()
            next_id = sample_next_token(outputs.logits[:, -1, :], args)
            cuda_sync()
            sample_s = time.perf_counter() - start
        sample_times.append(sample_s)
        generated_chunks.append(next_id)
        token_records.append(
            {
                "token_index": token_index,
                "source": "decode",
                "decode_model_s": round(decode_model_s, 6),
                "sample_s": round(sample_s, 6),
                "cpu_gap_before_model_s": round(cpu_gap_s, 6),
                "token_id": int(next_id.item()),
            }
        )
        last_gpu_done = time.perf_counter()
        if eos_ids and int(next_id.item()) in eos_ids:
            break

    with nvtx_range(args.nvtx, "concat_generated_tokens"):
        cuda_sync()
        start = time.perf_counter()
        generated = torch.cat(generated_chunks, dim=-1)
        cuda_sync()
        concat_s = time.perf_counter() - start

    report = build_timing_report(
        input_ids.shape[-1],
        len(token_records),
        prefill_s,
        decode_model_times,
        sample_times,
        cpu_gap_times,
        token_records,
    )
    report["concat_generated_tokens_s"] = round(concat_s, 6)
    return generated, report


def fast_generate(model, inputs, args):
    """低开销增量解码：避开 Transformers 通用 generate，减少 Python 侧通用逻辑。"""
    if args.static_cache:
        raise RuntimeError("--fast-decode 暂不支持 --static-cache，请使用默认动态 KV cache。")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_id = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = {int(item) for item in eos_id}
    elif eos_id is None:
        eos_ids = set()
    else:
        eos_ids = {int(eos_id)}

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = outputs.past_key_values
    next_id = sample_next_token(outputs.logits[:, -1, :], args)
    generated_chunks = [input_ids, next_id]

    if eos_ids and int(next_id.item()) in eos_ids:
        return torch.cat(generated_chunks, dim=-1)

    for _ in range(args.max_new_tokens - 1):
        outputs = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_id = sample_next_token(outputs.logits[:, -1, :], args)
        generated_chunks.append(next_id)
        if eos_ids and int(next_id.item()) in eos_ids:
            break
    return torch.cat(generated_chunks, dim=-1)


def build_timing_report(
    input_tokens,
    generated_tokens,
    prefill_s,
    decode_model_times,
    sample_times,
    cpu_gap_times,
    token_records,
):
    """构造 profiling JSON。"""
    decode_summary = summarize_seconds(decode_model_times)
    sample_summary = summarize_seconds(sample_times)
    cpu_gap_summary = summarize_seconds(cpu_gap_times)
    decode_total_s = decode_summary["total_s"] + sample_summary["total_s"]
    return {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "prefill_s": round(prefill_s, 6),
        "decode_model": decode_summary,
        "sample": sample_summary,
        "cpu_gap_before_decode_model": cpu_gap_summary,
        "decode_total_s": round(decode_total_s, 6),
        "tokens_per_second_decode_model_only": (
            round(max(generated_tokens - 1, 0) / decode_summary["total_s"], 3)
            if decode_summary["total_s"] > 0
            else None
        ),
        "tokens_per_second_decode_with_sampling": (
            round(generated_tokens / decode_total_s, 3) if decode_total_s > 0 else None
        ),
        "per_token": token_records,
    }


def generate_once(model, inputs, args, generate_kwargs):
    """按当前参数执行一次生成，profile 模式返回分阶段报告。"""
    if args.profile_timing:
        return profiled_generate(model, inputs, args)
    if args.fast_decode:
        return fast_generate(model, inputs, args), None
    return model.generate(**inputs, **generate_kwargs), None


@contextlib.contextmanager
def maybe_torch_profiler(profiler_dir):
    """按需启用 PyTorch profiler，并输出 TensorBoard trace。"""
    if not profiler_dir:
        yield None
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
    ) as profiler:
        yield profiler


def main(argv=None):
    """加载 Qwen/Qwen3.5-4B 并生成文本。"""
    args = parse_args(argv)
    prompt = resolve_prompt(args)
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False。")

        dtype = resolve_torch_dtype(args.dtype)
        device_map = {"": 0} if args.device == "cuda" else {"": "cpu"}

        log(f"开始加载 {MODEL_ID} ...")
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
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

        with nvtx_range(args.nvtx, "tokenize"):
            tokenize_start = time.perf_counter()
            inputs = build_inputs(tokenizer, prompt, args.disable_thinking)
            tokenize_elapsed = time.perf_counter() - tokenize_start

        with nvtx_range(args.nvtx, "input_to_device"):
            cuda_sync()
            input_to_device_start = time.perf_counter()
            inputs = inputs.to(model.device)
            cuda_sync()
            input_to_device_elapsed = time.perf_counter() - input_to_device_start

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": not args.greedy,
            "use_cache": True,
        }
        if args.static_cache:
            generate_kwargs["cache_implementation"] = "static"
        if not args.greedy:
            generate_kwargs["temperature"] = args.temperature

        warmup_runs = max(args.warmup_runs, 0)
        warmup_elapsed = 0.0
        if warmup_runs:
            log(f"开始预热，次数 {warmup_runs}，不计入正式推理耗时...")
            warmup_start = time.perf_counter()
            with torch.inference_mode():
                for run_index in range(warmup_runs):
                    with nvtx_range(args.nvtx, f"warmup_{run_index + 1}"):
                        warmup_generated, _ = generate_once(model, inputs, args, generate_kwargs)
                    del warmup_generated
                    cuda_sync()
            warmup_elapsed = time.perf_counter() - warmup_start
            log(f"预热完成，耗时 {warmup_elapsed:.2f}s")

        log("开始推理...")
        infer_start = time.perf_counter()
        timing_report = None
        with torch.inference_mode(), maybe_torch_profiler(args.torch_profiler) as profiler:
            with nvtx_range(args.nvtx, "generate"):
                generated, timing_report = generate_once(model, inputs, args, generate_kwargs)
            if profiler is not None:
                profiler.step()
        infer_elapsed = time.perf_counter() - infer_start
        log(f"推理完成，耗时 {infer_elapsed:.2f}s，max_new_tokens={args.max_new_tokens}")

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated[:, input_len:]
        with nvtx_range(args.nvtx, "text_decode"):
            text_decode_start = time.perf_counter()
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            text_decode_elapsed = time.perf_counter() - text_decode_start

        if timing_report is not None:
            timing_report["tokenize_s"] = round(tokenize_elapsed, 6)
            timing_report["input_to_device_s"] = round(input_to_device_elapsed, 6)
            timing_report["infer_wall_s"] = round(infer_elapsed, 6)
            timing_report["text_decode_s"] = round(text_decode_elapsed, 6)
            timing_report["warmup_runs"] = warmup_runs
            timing_report["warmup_s"] = round(warmup_elapsed, 6)
            timing_report["torch_profiler_dir"] = args.torch_profiler
            timing_report["nvtx_enabled"] = args.nvtx
            log("PROFILE_TIMING_JSON:")
            log(json.dumps(timing_report, ensure_ascii=False, indent=2))
        if args.torch_profiler:
            log(f"PyTorch profiler trace 已写入：{args.torch_profiler}")

        print(output_text)
        return 0
    except Exception as exc:
        print(f"推理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
