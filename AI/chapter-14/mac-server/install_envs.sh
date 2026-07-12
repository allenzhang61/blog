#!/usr/bin/env bash
set -euo pipefail

brew install localai

curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

mkdir -p "$HOME/.local/src"
if [ ! -d "$HOME/.local/src/sglang-metal/.git" ]; then
  git clone --depth 1 https://github.com/sgl-project/sglang.git "$HOME/.local/src/sglang-metal"
else
  git -C "$HOME/.local/src/sglang-metal" pull --ff-only
fi

cd "$HOME/.local/src/sglang-metal"
uv venv -p 3.12 sglang-metal
source sglang-metal/bin/activate
uv pip install --upgrade pip
uv run sgl-kernel/setup_metal.py install || true
rm -f python/pyproject.toml
mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"

cd -
mkdir -p AI/chapter-14/mac-server/localai/backends AI/chapter-14/mac-server/localai/models
env -u DEBUG local-ai backends install localai@metal-mlx \
  --backends-path "$PWD/AI/chapter-14/mac-server/localai/backends"
env -u DEBUG local-ai backends install localai@metal-llama-cpp \
  --backends-path "$PWD/AI/chapter-14/mac-server/localai/backends"

# The current metal-mlx backend bundle ships a transformers 5.x build that can
# fail during mlx_lm import on this machine. Pin to the latest 4.x line, which
# is enough for Qwen3-8B-MLX-4bit and keeps the backend importable.
if [ -x "$PWD/AI/chapter-14/mac-server/localai/backends/metal-mlx/venv/bin/python" ]; then
  "$PWD/AI/chapter-14/mac-server/localai/backends/metal-mlx/venv/bin/python" \
    -m pip install 'transformers<5'
fi
