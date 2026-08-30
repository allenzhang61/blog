#!/usr/bin/env bash
set -u

MODEL_DIR="${MODEL_DIR:-/home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf}"
BIN="${BIN:-./build-verify/local_llm}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-64}"
OUT_DIR="${OUT_DIR:-/tmp/local-llm-deepseek-quality-ablate}"

mkdir -p "${OUT_DIR}"

run_case() {
  local name="$1"
  local prompt="$2"
  shift 2
  local log="${OUT_DIR}/${name}.log"
  echo "===== ${name} ====="
  env \
    PROMPT="${prompt}" \
    LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1 \
    LOCAL_LLM_CUDA_DEQUANT_POOL_GB="${LOCAL_LLM_CUDA_DEQUANT_POOL_GB:-0.05}" \
    "$@" \
    "${BIN}" --model deepseek --model-dir "${MODEL_DIR}" --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
    >"${log}" 2>&1
  local code=$?
  echo "exit=${code} log=${log}"
  tail -12 "${log}"
}

FRANCE_PROMPT=$'User: What is the capital of France?\n\nAssistant:'
STORY_PROMPT='Write a short story about a robot learning to paint.'

run_case "preset_france" "${FRANCE_PROMPT}"
run_case "preset_story" "${STORY_PROMPT}"
run_case "prefill_q8_1_story" "${STORY_PROMPT}" \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_QUANT_DIRECT=1
run_case "safe_prefill_router_story" "${STORY_PROMPT}" \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_QUANT_DIRECT=1 \
  LOCAL_LLM_DEEPSEEK_PREFILL_QUANT_DIRECT_EXCLUDE_OPS=ds.gemm.router
run_case "safe_prefill_kvb_story" "${STORY_PROMPT}" \
  LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_QUANT_DIRECT=1 \
  LOCAL_LLM_DEEPSEEK_PREFILL_QUANT_DIRECT_EXCLUDE_OPS=ds.gemm.kv_b
