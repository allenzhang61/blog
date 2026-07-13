# Mac Studio 推理服务压测脚本

## 一键压测

```bash
chmod +x AI/chapter-14/mac-server/bench_mac_servers.sh
AI/chapter-14/mac-server/bench_mac_servers.sh
```

脚本会顺序压测本轮跑通的四个服务：

| 服务 | 模型名 |
| --- | --- |
| Ollama | `qwen3:8b` |
| LocalAI | `qwen3-8b-mlx` |
| SGLang MLX | `mlx-community/Qwen3-8B-4bit` |
| vLLM-Metal | `mlx-community/Qwen3-8B-4bit` |

默认输出 CSV 到：

```text
AI/chapter-14/mac-server/results/mac-server-bench-YYYYMMDD-HHMMSS.csv
```

CSV 列和文章里的最终压测结果表一致：

```csv
服务,并发,TTFT(ms),output tokens,total/wall_s,tokens/s,GPU active(%),avg_request_s,系统内存增量
```

可以通过环境变量调整输出路径、并发和 token 数：

```bash
OUT="AI/chapter-14/mac-server/results/mac-server-bench.csv" \
CONCURRENCY_LIST="1 2 4" MAX_TOKENS=128 CTX=2048 \
  AI/chapter-14/mac-server/bench_mac_servers.sh
```

只压测单个服务：

```bash
BENCH_SERVICES="Ollama" \
  AI/chapter-14/mac-server/bench_mac_servers.sh
```

如果要求必须采集 GPU 使用率，可以打开强制检查：

```bash
sudo -v
GPU_USAGE_REQUIRED=1 \
  AI/chapter-14/mac-server/bench_mac_servers.sh
```

没有可用 sudo 缓存时，脚本会直接报错退出。

也可以覆盖模型和端口：

```text
OLLAMA_MODEL=qwen3:8b
OLLAMA_MANAGED_SERVE=1
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_QUEUE=64
OLLAMA_KEEP_ALIVE=30m
OLLAMA_FLASH_ATTENTION=1
OLLAMA_CONTEXT_LENGTH=2048
LOCALAI_MODEL=qwen3-8b-mlx
SGLANG_MODEL=mlx-community/Qwen3-8B-4bit
VLLM_MODEL=mlx-community/Qwen3-8B-4bit
VLLM_MAX_MODEL_LEN=2048
VLLM_METAL_MEMORY_FRACTION=0.18
LOCALAI_PORT=18012
SGLANG_PORT=18011
VLLM_PORT=18010
```

脚本执行前会检查是否存在残留的 vLLM / SGLang / LocalAI / 压测进程。压测 Ollama、LocalAI、SGLang MLX、vLLM-Metal 时会自动启动服务，采集完成后自动停止。

LocalAI 会以压测专用轻量模式启动：关闭 Web UI、metrics、gallery、MCP、agent 等非必要功能，并设置 `--max-active-backends=1`，避免本轮单模型压测受到额外 backend 驻留影响。

`GPU active(%)` 通过 macOS `powermetrics` 采集，需要 sudo 权限。压测前可以先在终端执行：

```bash
sudo -v
```

然后再执行压测脚本。若没有 sudo 缓存，脚本会把 GPU 列填为 `未采集`，不会阻塞其他结果。

## 环境安装

已安装：

- LocalAI: `brew install localai`
- LocalAI backends: `localai@metal-mlx`、`localai@metal-llama-cpp`
- vLLM-Metal: `~/.venv-vllm-metal`
- SGLang MLX: `~/.local/src/sglang-metal/sglang-metal`

可复现安装命令：

```bash
AI/chapter-14/mac-server/install_envs.sh
```

## 服务说明

Ollama 默认由脚本以 tuned 参数启动终端版服务：

```bash
OLLAMA_NUM_PARALLEL=2 \
OLLAMA_MAX_QUEUE=64 \
OLLAMA_KEEP_ALIVE=30m \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_CONTEXT_LENGTH=2048 \
ollama serve
```

压测完成后脚本会恢复打开 Ollama 桌面 App。若想直接使用已经运行的桌面 App 服务，可以设置：

```text
OLLAMA_MANAGED_SERVE=0
```

默认设置了 `VLLM_MAX_MODEL_LEN=2048`、`VLLM_METAL_MEMORY_FRACTION=0.18`，避免 vLLM-Metal 按模型默认长上下文预留过大的 Metal/KV cache 内存。需要更大并发或更长上下文时可以手动调高：

```bash
VLLM_MAX_MODEL_LEN=4096 VLLM_METAL_MEMORY_FRACTION=0.35 \
  AI/chapter-14/mac-server/bench_mac_servers.sh
```

本机验证：原先未显式限制 `max-model-len` 时，vLLM-Metal 会使用模型默认 `40960`，在 `VLLM_METAL_MEMORY_FRACTION=0.65` 下预留约 `30.86GB` KV cache。改成 `VLLM_MAX_MODEL_LEN=2048`、`VLLM_METAL_MEMORY_FRACTION=0.18` 后，KV cache 降到约 `4.71GB`，日志显示对 2048 tokens/request 仍有约 `15.61x` 最大并发余量。

本轮四个服务都跑通的模型口径：

| 服务 | 模型名 |
| --- | --- |
| Ollama | `qwen3:8b` |
| vLLM-Metal | `mlx-community/Qwen3-8B-4bit` |
| SGLang MLX | `mlx-community/Qwen3-8B-4bit` |
| LocalAI | `qwen3-8b-mlx` |

注意：Ollama 使用自己的模型名，vLLM-Metal/SGLang/LocalAI 使用 Hugging Face/MLX 模型 id 或 LocalAI 模型配置，不能直接复用 Ollama 的模型名。

Ollama 的服务端调优参数需要在启动 `ollama serve` 前设置；如果使用桌面 App，通常不会继承当前 shell 的环境变量。本机 sweep 中，`OLLAMA_NUM_PARALLEL=2` 是当前较好的吞吐折中：

```bash
OLLAMA_NUM_PARALLEL=2 \
OLLAMA_MAX_QUEUE=64 \
OLLAMA_KEEP_ALIVE=30m \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_CONTEXT_LENGTH=2048 \
ollama serve
```

LocalAI 的 `metal-mlx` backend 当前可能需要把 backend venv 里的 `transformers` 固定到 4.x：

```bash
AI/chapter-14/mac-server/localai/backends/metal-mlx/venv/bin/python \
  -m pip install 'transformers<5'
```
