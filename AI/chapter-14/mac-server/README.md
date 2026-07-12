# Mac Studio 推理服务压测脚本

## Ollama

```bash
chmod +x AI/chapter-14/mac-server/bench_ollama.sh
AI/chapter-14/mac-server/bench_ollama.sh
```

默认可以压测本轮跑通的共同模型：

```text
qwen3:8b
```

可以通过环境变量调整：

```bash
MODEL="qwen3:8b" PREDICT=128 WARMUP_PREDICT=16 CTX=2048 \
  AI/chapter-14/mac-server/bench_ollama.sh
```

脚本每次只压测一个模型：先 warmup 一次，再记录第二次正式压测结果。

结果会写到：

```text
AI/chapter-14/mac-server/results/
```

## OpenAI-compatible server

用于压测 vLLM、SGLang、LocalAI 等兼容 `/v1/chat/completions` 的服务：

```bash
chmod +x AI/chapter-14/mac-server/bench_openai_compat.sh
BASE_URL="http://127.0.0.1:8000/v1" MODEL="your-model-name" \
  AI/chapter-14/mac-server/bench_openai_compat.sh
```

这个脚本同样每次只测一个模型，默认先发一次 16-token warmup 请求，再记录 128-token 正式请求。

如果服务需要 token：

```bash
API_KEY="xxx" BASE_URL="http://127.0.0.1:8000/v1" MODEL="your-model-name" \
  AI/chapter-14/mac-server/bench_openai_compat.sh
```

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

## 启动服务

Ollama 已由桌面 App 提供服务：

```text
http://127.0.0.1:11434
```

vLLM-Metal：

```bash
MODEL="mlx-community/Qwen3-8B-4bit" PORT=8000 \
  AI/chapter-14/mac-server/start_vllm_metal.sh
```

默认设置了 `VLLM_METAL_MEMORY_FRACTION=0.65`，避免 vLLM-Metal 在连续压测时抢占过多 Metal/KV cache 内存。需要更大并发或更长上下文时可以手动调高。

SGLang MLX：

```bash
MODEL="mlx-community/Qwen3-8B-4bit" PORT=8000 \
  AI/chapter-14/mac-server/start_sglang_mlx.sh
```

LocalAI：

```bash
PORT=8080 AI/chapter-14/mac-server/start_localai.sh
```

本轮四个服务都跑通的模型口径：

| 服务 | 模型名 |
| --- | --- |
| Ollama | `qwen3:8b` |
| vLLM-Metal | `mlx-community/Qwen3-8B-4bit` |
| SGLang MLX | `mlx-community/Qwen3-8B-4bit` |
| LocalAI | `qwen3-8b-mlx` |

注意：Ollama 使用自己的模型名，vLLM-Metal/SGLang/LocalAI 使用 Hugging Face/MLX 模型 id 或 LocalAI 模型配置，不能直接复用 Ollama 的模型名。

LocalAI 的 `metal-mlx` backend 当前可能需要把 backend venv 里的 `transformers` 固定到 4.x：

```bash
AI/chapter-14/mac-server/localai/backends/metal-mlx/venv/bin/python \
  -m pip install 'transformers<5'
```
