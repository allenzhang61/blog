import csv
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


BASE = os.environ.get("BASE", "http://127.0.0.1:8088/v1/chat/completions")
MODEL = os.environ.get("MODEL", "qwen3-8b-q4km")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "Qwen3-8B Q4_K_M GGUF")
SERVICE_LABEL = os.environ.get("SERVICE_LABEL", "llama.cpp CUDA")
PROMPT = "请用中文简要说明 LLM 推理中 prefill 和 decode 的区别。"
OUT = Path(
    os.environ.get(
        "OUT", r"D:\cuda-llm-bench\results\llama-cpp-cuda-qwen3-8b.csv"
    )
)
CONCURRENCIES = tuple(
    int(x.strip())
    for x in os.environ.get("CONCURRENCIES", "1,2,4").split(",")
    if x.strip()
)


def gpu_sample():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        util, mem, power = [x.strip() for x in out.split(",")]
        return float(util), float(mem), float(power)
    except Exception:
        return None


def sample_while(stop, samples):
    while not stop.is_set():
        sample = gpu_sample()
        if sample:
            samples.append(sample)
        time.sleep(0.2)


def make_payload(stream=False, suffix=""):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT + suffix}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": stream,
    }


def warmup():
    response = requests.post(BASE, json=make_payload(False, " warmup"), timeout=120)
    response.raise_for_status()


def measure_ttft():
    vals = []
    for i in range(5):
        start = time.perf_counter()
        with requests.post(
            BASE, json=make_payload(True, f" ttft-{i}"), stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    vals.append((time.perf_counter() - start) * 1000)
                    break
    vals.sort()
    return vals[len(vals) // 2]


def one_request(i):
    start = time.perf_counter()
    response = requests.post(BASE, json=make_payload(False, f" req-{i}"), timeout=180)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage") or {}
    completion = usage.get("completion_tokens") or 0
    total = usage.get("total_tokens") or 0
    return elapsed, completion, total


def run_concurrency(n, ttft):
    stop = threading.Event()
    samples = []
    sampler = threading.Thread(target=sample_while, args=(stop, samples), daemon=True)
    sampler.start()

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(one_request, i) for i in range(n)]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - start

    stop.set()
    sampler.join(timeout=1)

    output_tokens = sum(result[1] for result in results)
    avg_req = statistics.mean(result[0] for result in results)
    tokens_per_second = output_tokens / wall if wall > 0 else 0

    if samples:
        avg_gpu = statistics.mean(sample[0] for sample in samples)
        max_gpu = max(sample[0] for sample in samples)
        max_mem = max(sample[1] for sample in samples)
        avg_power = statistics.mean(sample[2] for sample in samples)
    else:
        avg_gpu = max_gpu = max_mem = avg_power = None

    return {
        "服务": SERVICE_LABEL,
        "模型": MODEL_LABEL,
        "并发": n,
        "TTFT(ms)": round(ttft, 1),
        "output tokens": output_tokens,
        "total/wall_s": round(wall, 2),
        "tokens/s": round(tokens_per_second, 1),
        "GPU avg(%)": round(avg_gpu, 1) if avg_gpu is not None else "未采集",
        "GPU max(%)": round(max_gpu, 1) if max_gpu is not None else "未采集",
        "VRAM max(MiB)": round(max_mem, 0) if max_mem is not None else "未采集",
        "Power avg(W)": round(avg_power, 1) if avg_power is not None else "未采集",
        "avg_request_s": round(avg_req, 2),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    warmup()
    ttft = measure_ttft()
    rows = [run_concurrency(n, ttft) for n in CONCURRENCIES]

    with OUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(OUT)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
