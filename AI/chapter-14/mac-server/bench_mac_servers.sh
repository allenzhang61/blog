#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/mac-server}"
OUT="${OUT:-$ROOT/results/mac-server-bench-$(date +%Y%m%d-%H%M%S).csv}"
PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"
CTX="${CTX:-2048}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TTFT_TOKENS="${TTFT_TOKENS:-16}"
WARMUP_TOKENS="${WARMUP_TOKENS:-4}"
TTFT_STREAM_WARMUPS="${TTFT_STREAM_WARMUPS:-1}"
TTFT_REPEATS="${TTFT_REPEATS:-5}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4}"
API_KEY="${API_KEY:-EMPTY}"
BENCH_SERVICES="${BENCH_SERVICES:-Ollama LocalAI SGLang MLX vLLM-Metal}"
GPU_SAMPLE_MS="${GPU_SAMPLE_MS:-200}"
GPU_USAGE_ENABLED="${GPU_USAGE_ENABLED:-auto}"
GPU_USAGE_REQUIRED="${GPU_USAGE_REQUIRED:-0}"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:8b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_NUM_BATCH="${OLLAMA_NUM_BATCH:-}"
OLLAMA_MANAGED_SERVE="${OLLAMA_MANAGED_SERVE:-1}"
OLLAMA_RESTORE_APP="${OLLAMA_RESTORE_APP:-1}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-64}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-$CTX}"
OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-}"

LOCALAI_MODEL="${LOCALAI_MODEL:-qwen3-8b-mlx}"
LOCALAI_HOST="${LOCALAI_HOST:-127.0.0.1}"
LOCALAI_PORT="${LOCALAI_PORT:-18012}"
LOCALAI_ROOT="${LOCALAI_ROOT:-$ROOT/localai}"

SGLANG_MODEL="${SGLANG_MODEL:-mlx-community/Qwen3-8B-4bit}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-18011}"

VLLM_MODEL="${VLLM_MODEL:-mlx-community/Qwen3-8B-4bit}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-18010}"
VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-0.18}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$VLLM_METAL_MEMORY_FRACTION}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$CTX}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"

mkdir -p "$(dirname "$OUT")"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

cleanup_pids=()
gpu_sampler_pid=""
gpu_sampler_file=""
managed_ollama_pid=""

cleanup() {
  local pid
  if [[ -n "${gpu_sampler_pid:-}" ]]; then
    kill -INT "$gpu_sampler_pid" 2>/dev/null || true
    wait "$gpu_sampler_pid" 2>/dev/null || true
    gpu_sampler_pid=""
  fi
  if [[ -n "${managed_ollama_pid:-}" ]]; then
    kill -INT "$managed_ollama_pid" 2>/dev/null || true
    wait "$managed_ollama_pid" 2>/dev/null || true
    managed_ollama_pid=""
  fi
  for pid in "${cleanup_pids[@]:-}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  sleep 2
  pkill -f 'powermetrics --samplers gpu_power' 2>/dev/null || true
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'sglang.launch_server' 2>/dev/null || true
  pkill -f 'local-ai run' 2>/dev/null || true
}
trap cleanup EXIT

ensure_no_residual_tasks() {
  local residual
  residual="$(
    ps -axo pid=,ppid=,command= \
      | awk -v self="$$" -v bashpid="${BASHPID:-$$}" -v parent="$PPID" '
        $1 == self || $1 == bashpid || $1 == parent { next }
        /vllm serve|sglang\.launch_server|local-ai run|powermetrics --samplers gpu_power/ {
          print
        }
      ' || true
  )"
  if [[ -n "$residual" ]]; then
    echo "Residual benchmark or serving process found:" >&2
    echo "$residual" >&2
    exit 1
  fi
}

system_used_mb() {
  local page_size total_bytes
  page_size="$(vm_stat | awk '/page size of/ {gsub(/[^0-9]/, "", $8); print $8; exit}')"
  total_bytes="$(sysctl -n hw.memsize)"
  vm_stat | awk -v page="$page_size" -v total="$total_bytes" '
    /Pages free:/ {gsub(/\./, "", $3); free=$3}
    /Pages speculative:/ {gsub(/\./, "", $3); speculative=$3}
    END {
      available=(free + speculative) * page;
      used=(total - available) / 1024 / 1024;
      printf "%.1f", used;
    }
  '
}

mb_to_gb() {
  awk -v mb="$1" 'BEGIN { printf "%.1f GB", mb / 1024 }'
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_csv_row() {
  local service="$1" concurrency="$2" ttft="$3" output_tokens="$4" wall="$5" tok_s="$6" gpu="$7" avg="$8" mem="$9"
  {
    csv_escape "$service"; printf ','
    printf '%s,' "$concurrency"
    printf '%s,' "$ttft"
    printf '%s,' "$output_tokens"
    printf '%s,' "$wall"
    printf '%s,' "$tok_s"
    csv_escape "$gpu"; printf ','
    printf '%s,' "$avg"
    csv_escape "$mem"
    printf '\n'
  } >> "$OUT"
}

gpu_sampling_available() {
  if [[ "$GPU_USAGE_ENABLED" == "0" || "$GPU_USAGE_ENABLED" == "false" ]]; then
    return 1
  fi
  command -v powermetrics >/dev/null 2>&1 || return 1
  sudo -n true >/dev/null 2>&1 || return 1
}

start_gpu_sampler() {
  gpu_sampler_pid=""
  gpu_sampler_file=""
  if ! gpu_sampling_available; then
    if [[ "$GPU_USAGE_REQUIRED" == "1" || "$GPU_USAGE_REQUIRED" == "true" ]]; then
      echo "GPU usage sampling requires non-interactive sudo. Run 'sudo -v' in the same shell, or set GPU_USAGE_REQUIRED=0." >&2
      exit 1
    fi
    return 0
  fi
  gpu_sampler_file="$(mktemp)"
  sudo -n powermetrics --samplers gpu_power -i "$GPU_SAMPLE_MS" >"$gpu_sampler_file" 2>&1 &
  gpu_sampler_pid=$!
  sleep "$(awk -v ms="$GPU_SAMPLE_MS" 'BEGIN { printf "%.3f", (ms / 1000) * 1.5 }')"
}

stop_gpu_sampler() {
  local value
  if [[ -z "${gpu_sampler_pid:-}" || -z "${gpu_sampler_file:-}" ]]; then
    printf '未采集'
    return 0
  fi
  kill -INT "$gpu_sampler_pid" 2>/dev/null || true
  wait "$gpu_sampler_pid" 2>/dev/null || true
  gpu_sampler_pid=""
  value="$(
    perl -ne '
      while (/(?:GPU active residency|GPU Active residency|GPU residency|GPU HW active residency).*?([0-9]+(?:\.[0-9]+)?)%/gi) {
        push @values, $1;
      }
      END {
        if (@values) {
          my $sum = 0;
          $sum += $_ for @values;
          printf "%.1f%%", $sum / @values;
        } else {
          print "未采集";
        }
      }
    ' "$gpu_sampler_file"
  )"
  rm -f "$gpu_sampler_file"
  gpu_sampler_file=""
  printf '%s' "$value"
}

openai_payload() {
  local model="$1" max_tokens="$2" stream="$3" prompt="$4"
  jq -n \
    --arg model "$model" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    --argjson stream "$stream" \
    '{
      model: $model,
      messages: [{role: "user", content: $prompt}],
      max_tokens: $max_tokens,
      temperature: 0,
      stream: $stream
    }'
}

ollama_payload() {
  local model="$1" predict="$2" stream="$3" prompt="$4"
  jq -n \
    --arg model "$model" \
    --arg prompt "$prompt" \
    --argjson predict "$predict" \
    --argjson ctx "$CTX" \
    --argjson stream "$stream" \
    --arg num_batch "$OLLAMA_NUM_BATCH" \
    '{
      model: $model,
      prompt: $prompt,
      stream: $stream,
      options: {
        num_predict: $predict,
        num_ctx: $ctx,
        temperature: 0
      }
    }
    | if $num_batch != "" then .options.num_batch = ($num_batch | tonumber) else . end'
}

run_openai() {
  local base_url="$1" model="$2" max_tokens="$3" stream="$4" prompt="$5"
  curl -fsS -N --max-time 300 "$base_url/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' \
    -d "$(openai_payload "$model" "$max_tokens" "$stream" "$prompt")"
}

run_ollama() {
  local model="$1" predict="$2" stream="$3" prompt="$4"
  curl -fsS -N --max-time 300 "$OLLAMA_HOST/api/generate" \
    -H 'Content-Type: application/json' \
    -d "$(ollama_payload "$model" "$predict" "$stream" "$prompt")"
}

wait_openai_ready() {
  local base_url="$1"
  local i
  for i in {1..90}; do
    if curl -fsS --max-time 2 "$base_url/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_vllm() {
  local args
  export VLLM_METAL_MEMORY_FRACTION
  args=(
    serve "$VLLM_MODEL"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  )
  if [[ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]]; then
    args+=(--kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
  fi
  source "$HOME/.venv-vllm-metal/bin/activate"
  exec vllm "${args[@]}"
}

start_sglang() {
  source "$HOME/.local/src/sglang-metal/sglang-metal/bin/activate"
  export SGLANG_USE_MLX="${SGLANG_USE_MLX:-1}"
  exec python -m sglang.launch_server \
    --model-path "$SGLANG_MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    --disable-cuda-graph
}

start_localai() {
  mkdir -p "$LOCALAI_ROOT/models" "$LOCALAI_ROOT/backends"
  exec env -u DEBUG local-ai run \
    --address "$LOCALAI_HOST:$LOCALAI_PORT" \
    --models-path "$LOCALAI_ROOT/models" \
    --backends-path "$LOCALAI_ROOT/backends" \
    --max-active-backends=1 \
    --disable-web-ui \
    --disable-metrics-endpoint \
    --disable-gallery-endpoint \
    --disable-mcp \
    --disable-agents \
    --disable-local-ai-assistant
}

stop_existing_ollama() {
  osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
  pkill -TERM -f '/Applications/Ollama.app/Contents/Resources/ollama serve' 2>/dev/null || true
  pkill -TERM -f '/Applications/Ollama.app/Contents/MacOS/Ollama' 2>/dev/null || true
  pkill -TERM -f '^ollama serve$' 2>/dev/null || true
  sleep 2
  pkill -KILL -f '/Applications/Ollama.app/Contents/Resources/ollama serve' 2>/dev/null || true
  pkill -KILL -f '/Applications/Ollama.app/Contents/MacOS/Ollama' 2>/dev/null || true
  pkill -KILL -f '^ollama serve$' 2>/dev/null || true
}

wait_ollama_ready() {
  local i
  for i in {1..45}; do
    if curl -fsS --max-time 2 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_managed_ollama() {
  local log
  if [[ "$OLLAMA_MANAGED_SERVE" != "1" && "$OLLAMA_MANAGED_SERVE" != "true" ]]; then
    wait_ollama_ready
    return
  fi

  stop_existing_ollama
  log="$ROOT/results/Ollama-managed-server.log"
  (
    export OLLAMA_NUM_PARALLEL
    export OLLAMA_MAX_QUEUE
    export OLLAMA_KEEP_ALIVE
    export OLLAMA_FLASH_ATTENTION
    export OLLAMA_CONTEXT_LENGTH
    if [[ -n "$OLLAMA_KV_CACHE_TYPE" ]]; then
      export OLLAMA_KV_CACHE_TYPE
    fi
    exec ollama serve
  ) >"$log" 2>&1 &
  managed_ollama_pid=$!

  if ! wait_ollama_ready; then
    echo "Managed Ollama failed to become ready" >&2
    tail -n 80 "$log" >&2 || true
    exit 1
  fi
}

stop_managed_ollama() {
  if [[ -n "${managed_ollama_pid:-}" ]]; then
    kill -INT "$managed_ollama_pid" 2>/dev/null || true
    sleep 2
    kill -TERM "$managed_ollama_pid" 2>/dev/null || true
    managed_ollama_pid=""
  fi
  if [[ "$OLLAMA_MANAGED_SERVE" == "1" || "$OLLAMA_MANAGED_SERVE" == "true" ]]; then
    stop_existing_ollama
    if [[ "$OLLAMA_RESTORE_APP" == "1" || "$OLLAMA_RESTORE_APP" == "true" ]]; then
      open -a Ollama >/dev/null 2>&1 || true
    fi
  fi
}

measure_ttft_openai() {
  local base_url="$1" model="$2"
  local i
  run_openai "$base_url" "$model" "$WARMUP_TOKENS" false "$PROMPT" >/dev/null
  for i in $(seq 1 "$TTFT_STREAM_WARMUPS"); do
    curl -fsS -N --max-time 300 \
      -o /dev/null \
      "$base_url/chat/completions" \
      -H "Authorization: Bearer $API_KEY" \
      -H 'Content-Type: application/json' \
      -d "$(openai_payload "$model" "$TTFT_TOKENS" true "$PROMPT")"
  done
  for i in $(seq 1 "$TTFT_REPEATS"); do
    curl -fsS -N --max-time 300 \
      -o /dev/null \
      -w '%{time_starttransfer}\n' \
      "$base_url/chat/completions" \
      -H "Authorization: Bearer $API_KEY" \
      -H 'Content-Type: application/json' \
      -d "$(openai_payload "$model" "$TTFT_TOKENS" true "${PROMPT} 第 ${i} 次。")"
  done | median
}

measure_ttft_ollama() {
  local i
  run_ollama "$OLLAMA_MODEL" "$WARMUP_TOKENS" false "$PROMPT" >/dev/null
  for i in $(seq 1 "$TTFT_STREAM_WARMUPS"); do
    curl -fsS -N --max-time 300 \
      -o /dev/null \
      "$OLLAMA_HOST/api/generate" \
      -H 'Content-Type: application/json' \
      -d "$(ollama_payload "$OLLAMA_MODEL" "$TTFT_TOKENS" true "$PROMPT")"
  done
  for i in $(seq 1 "$TTFT_REPEATS"); do
    curl -fsS -N --max-time 300 \
      -o /dev/null \
      -w '%{time_starttransfer}\n' \
      "$OLLAMA_HOST/api/generate" \
      -H 'Content-Type: application/json' \
      -d "$(ollama_payload "$OLLAMA_MODEL" "$TTFT_TOKENS" true "${PROMPT} 第 ${i} 次。")"
  done | median
}

bench_openai_concurrency() {
  local base_url="$1" model="$2" concurrency="$3"
  local tmpdir start_ns end_ns wall_s i
  tmpdir="$(mktemp -d)"
  start_ns="$(date +%s%N)"
  for i in $(seq 1 "$concurrency"); do
    (
      local prompt response req_start req_end elapsed
      prompt="${PROMPT}"$'\n\n'"请求编号：${i}。"
      req_start="$(date +%s%N)"
      response="$(run_openai "$base_url" "$model" "$MAX_TOKENS" false "$prompt")"
      req_end="$(date +%s%N)"
      elapsed="$(awk -v s="$req_start" -v e="$req_end" 'BEGIN { printf "%.6f", (e - s) / 1000000000 }')"
      jq -n --arg elapsed "$elapsed" --argjson response "$response" \
        '{elapsed_s: ($elapsed | tonumber), response: $response}' > "$tmpdir/$i.json"
    ) &
  done
  wait
  end_ns="$(date +%s%N)"
  wall_s="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f", (e - s) / 1000000000 }')"
  jq -s -r --arg wall "$wall_s" '
    def out: .response.usage.completion_tokens // 0;
    [
      (map(out) | add),
      $wall,
      (if ($wall | tonumber) > 0 then ((map(out) | add) / ($wall | tonumber)) else 0 end),
      ((map(.elapsed_s) | add) / length)
    ] | @tsv
  ' "$tmpdir"/*.json
  rm -rf "$tmpdir"
}

bench_ollama_concurrency() {
  local concurrency="$1"
  local tmpdir start_ns end_ns wall_s i
  tmpdir="$(mktemp -d)"
  start_ns="$(date +%s%N)"
  for i in $(seq 1 "$concurrency"); do
    (
      local prompt response req_start req_end elapsed
      prompt="${PROMPT}"$'\n\n'"请求编号：${i}。"
      req_start="$(date +%s%N)"
      response="$(run_ollama "$OLLAMA_MODEL" "$MAX_TOKENS" false "$prompt")"
      req_end="$(date +%s%N)"
      elapsed="$(awk -v s="$req_start" -v e="$req_end" 'BEGIN { printf "%.6f", (e - s) / 1000000000 }')"
      jq -n --arg elapsed "$elapsed" --argjson response "$response" \
        '{elapsed_s: ($elapsed | tonumber), response: $response}' > "$tmpdir/$i.json"
    ) &
  done
  wait
  end_ns="$(date +%s%N)"
  wall_s="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f", (e - s) / 1000000000 }')"
  jq -s -r --arg wall "$wall_s" '
    def out: .response.eval_count // 0;
    [
      (map(out) | add),
      $wall,
      (if ($wall | tonumber) > 0 then ((map(out) | add) / ($wall | tonumber)) else 0 end),
      ((map(.elapsed_s) | add) / length)
    ] | @tsv
  ' "$tmpdir"/*.json
  rm -rf "$tmpdir"
}

round2() {
  awk -v n="$1" 'BEGIN { printf "%.2f", n }'
}

round3() {
  awk -v n="$1" 'BEGIN { printf "%.3f", n }'
}

seconds_to_ms() {
  awk -v n="$1" 'BEGIN { printf "%.1f", n * 1000 }'
}

round1() {
  awk -v n="$1" 'BEGIN { printf "%.1f", n }'
}

median() {
  sort -n | awk '
    { values[NR] = $1 }
    END {
      if (NR == 0) {
        exit 1
      }
      if (NR % 2 == 1) {
        printf "%.6f", values[(NR + 1) / 2]
      } else {
        printf "%.6f", (values[NR / 2] + values[NR / 2 + 1]) / 2
      }
    }'
}

write_header() {
  printf '服务,并发,TTFT(ms),output tokens,total/wall_s,tokens/s,GPU active(%%),avg_request_s,系统内存增量\n' > "$OUT"
}

should_bench_service() {
  local service="$1"
  case " $BENCH_SERVICES " in
    *" $service "*) return 0 ;;
    *) return 1 ;;
  esac
}

bench_ollama_service() {
  echo "==> Ollama" >&2
  start_managed_ollama
  ollama stop "$OLLAMA_MODEL" >/dev/null 2>&1 || true
  sleep 3
  local before after delta mem ttft c metrics out wall tok gpu avg
  before="$(system_used_mb)"
  run_ollama "$OLLAMA_MODEL" "$WARMUP_TOKENS" false "$PROMPT" >/dev/null
  after="$(system_used_mb)"
  delta="$(awk -v b="$before" -v a="$after" 'BEGIN { printf "%.1f", a - b }')"
  mem="$(mb_to_gb "$delta")"
  ttft="$(seconds_to_ms "$(measure_ttft_ollama)")"
  for c in $CONCURRENCY_LIST; do
    start_gpu_sampler
    metrics="$(bench_ollama_concurrency "$c")"
    gpu="$(stop_gpu_sampler)"
    out="$(awk '{print $1}' <<<"$metrics")"
    wall="$(round2 "$(awk '{print $2}' <<<"$metrics")")"
    tok="$(round1 "$(awk '{print $3}' <<<"$metrics")")"
    avg="$(round2 "$(awk '{print $4}' <<<"$metrics")")"
    append_csv_row "Ollama" "$c" "$ttft" "$out" "$wall" "$tok" "$gpu" "$avg" "$mem"
  done
  stop_managed_ollama
}

bench_openai_service() {
  local service="$1" model="$2" base_url="$3" start_fn="$4"
  echo "==> $service" >&2
  ensure_no_residual_tasks
  local before after delta mem ttft c metrics out wall tok gpu avg srv
  before="$(system_used_mb)"
  "$start_fn" >"$ROOT/results/${service// /-}-server.log" 2>&1 &
  srv=$!
  cleanup_pids+=("$srv")
  if ! wait_openai_ready "$base_url"; then
    echo "$service failed to become ready" >&2
    tail -n 80 "$ROOT/results/${service// /-}-server.log" >&2 || true
    exit 1
  fi
  run_openai "$base_url" "$model" "$WARMUP_TOKENS" false "$PROMPT" >/dev/null
  after="$(system_used_mb)"
  delta="$(awk -v b="$before" -v a="$after" 'BEGIN { printf "%.1f", a - b }')"
  mem="$(mb_to_gb "$delta")"
  ttft="$(seconds_to_ms "$(measure_ttft_openai "$base_url" "$model")")"
  for c in $CONCURRENCY_LIST; do
    start_gpu_sampler
    metrics="$(bench_openai_concurrency "$base_url" "$model" "$c")"
    gpu="$(stop_gpu_sampler)"
    out="$(awk '{print $1}' <<<"$metrics")"
    wall="$(round2 "$(awk '{print $2}' <<<"$metrics")")"
    tok="$(round1 "$(awk '{print $3}' <<<"$metrics")")"
    avg="$(round2 "$(awk '{print $4}' <<<"$metrics")")"
    append_csv_row "$service" "$c" "$ttft" "$out" "$wall" "$tok" "$gpu" "$avg" "$mem"
  done
  kill -INT "$srv" 2>/dev/null || true
  sleep 4
  cleanup_pids=("${cleanup_pids[@]/$srv}")
  cleanup
}

write_header
ensure_no_residual_tasks
if should_bench_service "Ollama"; then
  bench_ollama_service
fi
if should_bench_service "LocalAI"; then
  bench_openai_service "LocalAI" "$LOCALAI_MODEL" "http://$LOCALAI_HOST:$LOCALAI_PORT/v1" start_localai
fi
if should_bench_service "SGLang MLX"; then
  bench_openai_service "SGLang MLX" "$SGLANG_MODEL" "http://$SGLANG_HOST:$SGLANG_PORT/v1" start_sglang
fi
if should_bench_service "vLLM-Metal"; then
  bench_openai_service "vLLM-Metal" "$VLLM_MODEL" "http://$VLLM_HOST:$VLLM_PORT/v1" start_vllm
fi

echo "$OUT"
