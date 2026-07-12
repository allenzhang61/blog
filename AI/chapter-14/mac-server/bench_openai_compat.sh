#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-}"
API_KEY="${API_KEY:-EMPTY}"
OUT="${OUT:-AI/chapter-14/mac-server/results/openai-compat-$(date +%Y%m%d-%H%M%S).tsv}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_TOKENS="${WARMUP_TOKENS:-16}"
PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required, for example: MODEL=mlx-community/Qwen3-8B-4bit $0" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

payload="$(
  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{
      model: $model,
      messages: [
        {role: "user", content: $prompt}
      ],
      max_tokens: $max_tokens,
      temperature: 0,
      stream: false
    }'
)"

warmup_payload="$(
  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$WARMUP_TOKENS" \
    '{
      model: $model,
      messages: [
        {role: "user", content: $prompt}
      ],
      max_tokens: $max_tokens,
      temperature: 0,
      stream: false
    }'
)"

printf "model\tbase_url\tmax_tokens\ttotal_s\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tcompletion_tok_s\n" > "$OUT"

echo "==> warmup: $MODEL" >&2
curl -fsS --max-time 300 "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "$warmup_payload" >/dev/null

echo "==> benchmark: $MODEL" >&2
start_ns="$(date +%s%N)"
response="$(
  curl -fsS --max-time 180 "$BASE_URL/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' \
    -d "$payload"
)"
end_ns="$(date +%s%N)"

elapsed_s="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.6f", (end - start) / 1000000000 }')"

jq -r \
  --arg model "$MODEL" \
  --arg base_url "$BASE_URL" \
  --arg max_tokens "$MAX_TOKENS" \
  --arg elapsed "$elapsed_s" \
  '[
    $model,
    $base_url,
    $max_tokens,
    $elapsed,
    (.usage.prompt_tokens // 0),
    (.usage.completion_tokens // 0),
    (.usage.total_tokens // 0),
    (if ($elapsed | tonumber) > 0 then ((.usage.completion_tokens // 0) / ($elapsed | tonumber)) else 0 end)
  ] | @tsv' <<<"$response" >> "$OUT"

echo "$OUT"
