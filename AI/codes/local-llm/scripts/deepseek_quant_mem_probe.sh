#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BINARY="${LOCAL_LLM_BINARY:-$ROOT_DIR/build-verify/local_llm}"
MODEL="${LOCAL_LLM_DEEPSEEK_MODEL:-/home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/local-llm-deepseek-quant-mem-probe}"
DEQUANT_POOL_GB="${LOCAL_LLM_CUDA_DEQUANT_POOL_GB:-0.05}"

FRANCE_PROMPT="${FRANCE_PROMPT:-User: What is the capital of France?

Assistant:}"
STORY_PROMPT="${STORY_PROMPT:-User: Write a long story about a robot exploring Mars.

Assistant:}"

mkdir -p "$OUT_DIR"

echo "== Environment =="
echo "ROOT_DIR=$ROOT_DIR"
echo "BINARY=$BINARY"
echo "MODEL=$MODEL"
echo "OUT_DIR=$OUT_DIR"
echo "LOCAL_LLM_CUDA_DEQUANT_POOL_GB=$DEQUANT_POOL_GB"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
fi
if [[ -f "$MODEL" ]]; then
  ls -lh "$MODEL"
  du -b "$MODEL"
fi
echo

run_case() {
  local name="$1"
  local prompt="$2"
  local tokens="$3"
  local profile="$4"
  shift 4

  local case_dir="$OUT_DIR/$name"
  local log="$OUT_DIR/$name.log"
  mkdir -p "$case_dir"

  echo "== Case: $name =="
  echo "tokens=$tokens profile=$profile"
  echo "env=$*"

  if [[ "$profile" == "1" ]]; then
    env "$@" \
      PROMPT="$prompt" \
      LOCAL_LLM_CUDA_DEQUANT_POOL_GB="$DEQUANT_POOL_GB" \
      "$BINARY" --model deepseek --model-dir "$MODEL" \
      --max-output-tokens "$tokens" \
      --profile --profile-dir "$case_dir" --profile-sample-every 16 \
      >"$log" 2>&1
  else
    env "$@" \
      PROMPT="$prompt" \
      LOCAL_LLM_CUDA_DEQUANT_POOL_GB="$DEQUANT_POOL_GB" \
      "$BINARY" --model deepseek --model-dir "$MODEL" \
      --max-output-tokens "$tokens" \
      >"$log" 2>&1
  fi
  local status=$?
  echo "exit=$status"

  grep -n "生成结果\|tokens_per_sec" "$log" || true
  grep -n "out of memory\|cudaMalloc s_weight" "$log" || true

  local summary="$case_dir/profile_deepseek_summary.md"
  if [[ -f "$summary" ]]; then
    grep -n "prefill\|device used\|上传总量\|驻留峰值\|dequantize_q\|ds.gemm.e_swiglu\|ds.gemm.s_swiglu" "$summary" | head -80 || true
  else
    echo "summary=missing"
  fi
  echo "log=$log"
  echo
}

run_case "stage2_france_profile" "$FRANCE_PROMPT" 16 1 \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1

run_case "stage3_france_profile" "$FRANCE_PROMPT" 16 1 \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1 \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_QUANT_DIRECT=1

run_case "stage2_story_128_noprofile" "$STORY_PROMPT" 128 0 \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1

run_case "stage3_story_128_noprofile" "$STORY_PROMPT" 128 0 \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1 \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_QUANT_DIRECT=1
