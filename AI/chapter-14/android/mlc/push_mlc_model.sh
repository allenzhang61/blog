#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/android}"
PACKAGE="${PACKAGE:-ai.mlc.mlcengineexample}"
MODEL_ID="${MODEL_ID:-Qwen2.5-1.5B-Instruct-q4f16_1-MLC}"
MODEL_REPO="${MODEL_REPO:-mlc-ai/$MODEL_ID}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT/downloads/mlc-models/$MODEL_ID}"
DEVICE_DIR="${DEVICE_DIR:-/data/data/$PACKAGE/files/$MODEL_ID}"
HF_BIN="${HF_BIN:-$ROOT/build/mlc-venv15/bin/hf}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

require_cmd adb
require_cmd tar

if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="hf"
  require_cmd "$HF_BIN"
fi

mkdir -p "$LOCAL_DIR"
"$HF_BIN" download "$MODEL_REPO" --local-dir "$LOCAL_DIR"

adb shell "run-as '$PACKAGE' mkdir -p 'files/$MODEL_ID'"
tar -C "$(dirname "$LOCAL_DIR")" -cf - "$(basename "$LOCAL_DIR")" \
  | adb exec-in run-as "$PACKAGE" tar -C files -xf -

adb shell "run-as '$PACKAGE' ls '$DEVICE_DIR/mlc-chat-config.json'"
echo "$DEVICE_DIR"
