#!/usr/bin/env bash
set -euo pipefail

# SGLang MLX uses Hugging Face / MLX model ids or local model paths.
# Override MODEL with a model supported by SGLang's MLX backend.
MODEL="${MODEL:-mlx-community/Qwen3-8B-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

source "$HOME/.local/src/sglang-metal/sglang-metal/bin/activate"

export SGLANG_USE_MLX="${SGLANG_USE_MLX:-1}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --disable-cuda-graph
