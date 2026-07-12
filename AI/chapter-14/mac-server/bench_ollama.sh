#!/usr/bin/env bash
set -euo pipefail

HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL="${MODEL:-qwen3:8b}"
OUT="${OUT:-AI/chapter-14/mac-server/results/ollama-$(date +%Y%m%d-%H%M%S).tsv}"
PREDICT="${PREDICT:-128}"
WARMUP_PREDICT="${WARMUP_PREDICT:-16}"
CTX="${CTX:-2048}"
PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"

mkdir -p "$(dirname "$OUT")"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

if ! curl -fsS --max-time 3 "$HOST/api/tags" >/dev/null; then
  echo "Ollama API is not reachable: $HOST" >&2
  exit 1
fi

mem_mb() {
  ps -axo rss,command \
    | awk '/[O]llama|[o]llama serve|[o]llama runner/ {sum += $1} END {printf "%.1f", sum / 1024}'
}

generate_payload() {
  local model="$1"
  local prompt="$2"
  local predict="$3"
  jq -n \
    --arg model "$model" \
    --arg prompt "$prompt" \
    --argjson predict "$predict" \
    --argjson ctx "$CTX" \
    '{
      model: $model,
      prompt: $prompt,
      stream: false,
      options: {
        num_predict: $predict,
        num_ctx: $ctx,
        temperature: 0
      }
    }'
}

run_once() {
  local model="$1"
  local predict="$2"
  local payload
  payload="$(generate_payload "$model" "$PROMPT" "$predict")"
  curl -fsS --max-time 180 "$HOST/api/generate" \
    -H 'Content-Type: application/json' \
    -d "$payload"
}

printf "model\tcontext\tpredict\ttotal_s\tload_s\tprompt_tokens\tprompt_s\tprompt_tok_s\toutput_tokens\toutput_s\toutput_tok_s\tollama_rss_mb\n" > "$OUT"

echo "==> warmup: $MODEL" >&2
run_once "$MODEL" "$WARMUP_PREDICT" >/dev/null

echo "==> benchmark: $MODEL" >&2
before_mem="$(mem_mb)"
result="$(run_once "$MODEL" "$PREDICT")"
after_mem="$(mem_mb)"

jq -r \
  --arg model "$MODEL" \
  --arg ctx "$CTX" \
  --arg predict "$PREDICT" \
  --arg mem "$after_mem" \
  '[
    $model,
    $ctx,
    $predict,
    ((.total_duration // 0) / 1000000000),
    ((.load_duration // 0) / 1000000000),
    (.prompt_eval_count // 0),
    ((.prompt_eval_duration // 0) / 1000000000),
    (if (.prompt_eval_duration // 0) > 0 then ((.prompt_eval_count // 0) / ((.prompt_eval_duration // 0) / 1000000000)) else 0 end),
    (.eval_count // 0),
    ((.eval_duration // 0) / 1000000000),
    (if (.eval_duration // 0) > 0 then ((.eval_count // 0) / ((.eval_duration // 0) / 1000000000)) else 0 end),
    $mem
  ] | @tsv' <<<"$result" >> "$OUT"

echo "    memory before/after MB: $before_mem -> $after_mem" >&2

echo "$OUT"
