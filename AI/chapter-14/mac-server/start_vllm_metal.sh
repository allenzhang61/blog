#!/usr/bin/env bash
set -euo pipefail

# vLLM-Metal uses Hugging Face / MLX model ids or local model paths.
# Override MODEL with a model supported by vllm-metal.
MODEL="${MODEL:-mlx-community/Qwen3-8B-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-0.65}"
export VLLM_METAL_MEMORY_FRACTION

source "$HOME/.venv-vllm-metal/bin/activate"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT"
