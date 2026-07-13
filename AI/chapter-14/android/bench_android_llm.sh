#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/android}"
OUT="${OUT:-$ROOT/results/pixel9-bench-$(date +%Y%m%d-%H%M%S).csv}"

SERVICE="${SERVICE:-Android LLM Server}"
API_STYLE="${API_STYLE:-auto}" # auto | openai | ollama
DEVICE_PORT="${DEVICE_PORT:-8080}"
HOST_PORT="${HOST_PORT:-18080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$HOST_PORT}"
MODEL="${MODEL:-local-model}"
ANDROID_PACKAGE="${ANDROID_PACKAGE:-}"
ANDROID_PROCESS_NAME="${ANDROID_PROCESS_NAME:-}"
ANDROID_GPU_UID="${ANDROID_GPU_UID:-}"

PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TTFT_TOKENS="${TTFT_TOKENS:-16}"
WARMUP_TOKENS="${WARMUP_TOKENS:-4}"
TTFT_REPEATS="${TTFT_REPEATS:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
API_KEY="${API_KEY:-EMPTY}"

mkdir -p "$(dirname "$OUT")"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

require_cmd adb
require_cmd curl
require_cmd jq
require_cmd awk

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_csv_row() {
  local service="$1" api="$2" model="$3" ttft="$4" output_tokens="$5" wall="$6" tok_s="$7" avg="$8" cpu="$9" gpu="${10}" gpu_freq="${11}" proc_mem="${12}" mem_delta="${13}" battery="${14}" thermal="${15}"
  {
    csv_escape "$service"; printf ','
    csv_escape "$api"; printf ','
    csv_escape "$model"; printf ','
    printf '%s,' "$ttft"
    printf '%s,' "$output_tokens"
    printf '%s,' "$wall"
    printf '%s,' "$tok_s"
    printf '%s,' "$avg"
    csv_escape "$cpu"; printf ','
    csv_escape "$gpu"; printf ','
    csv_escape "$gpu_freq"; printf ','
    csv_escape "$proc_mem"; printf ','
    csv_escape "$mem_delta"; printf ','
    csv_escape "$battery"; printf ','
    csv_escape "$thermal"
    printf '\n'
  } >> "$OUT"
}

device_count() {
  adb devices | awk 'NR > 1 && $2 == "device" { n++ } END { print n + 0 }'
}

ensure_device() {
  local count
  count="$(device_count)"
  if [[ "$count" -eq 0 ]]; then
    echo "No authorized Android device found. Enable USB debugging and confirm the RSA prompt on the phone." >&2
    adb devices -l >&2 || true
    exit 1
  fi
}

adb_shell() {
  adb shell "$@" 2>/dev/null | tr -d '\r'
}

mem_available_kb() {
  adb_shell "awk '/MemAvailable:/ {print \$2}' /proc/meminfo" | awk 'NF {print $1; exit}'
}

device_used_mb() {
  local total avail
  total="$(adb_shell "awk '/MemTotal:/ {print \$2}' /proc/meminfo" | awk 'NF {print $1; exit}')"
  avail="$(mem_available_kb)"
  if [[ -z "${total:-}" || -z "${avail:-}" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v total="$total" -v avail="$avail" 'BEGIN { printf "%.1f", (total - avail) / 1024 }'
}

mb_delta_to_gb() {
  local before="$1" after="$2"
  if [[ "$before" == "未采集" || "$after" == "未采集" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v before="$before" -v after="$after" 'BEGIN { printf "%.1f GB", (after - before) / 1024 }'
}

package_pss() {
  if [[ -z "$ANDROID_PACKAGE" ]]; then
    printf '未采集'
    return 0
  fi
  local kb
  kb="$(adb_shell "dumpsys meminfo '$ANDROID_PACKAGE' | awk '/TOTAL PSS:/ {print \$3; exit} /^ *TOTAL / {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "${kb:-}" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v kb="$kb" 'BEGIN { printf "%.1f MB", kb / 1024 }'
}

process_rss() {
  if [[ -z "$ANDROID_PROCESS_NAME" ]]; then
    printf '未采集'
    return 0
  fi
  local kb
  kb="$(adb_shell "ps -A -o NAME,RSS | awk '\$1 == \"$ANDROID_PROCESS_NAME\" {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "${kb:-}" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v kb="$kb" 'BEGIN { printf "%.1f MB", kb / 1024 }'
}

process_memory() {
  local value
  value="$(package_pss)"
  if [[ "$value" != "未采集" ]]; then
    printf '%s' "$value"
    return 0
  fi
  process_rss
}

process_pid() {
  if [[ -z "$ANDROID_PROCESS_NAME" ]]; then
    printf ''
    return 0
  fi
  adb_shell "pidof '$ANDROID_PROCESS_NAME' 2>/dev/null || ps -A -o PID,NAME | awk '\$2 == \"$ANDROID_PROCESS_NAME\" {print \$1; exit}'" \
    | awk 'NF {print $1; exit}'
}

cpu_count() {
  adb_shell "grep -c '^processor' /proc/cpuinfo" | awk 'NF {print $1; exit}'
}

cpu_total_jiffies() {
  adb_shell "awk '/^cpu / {sum=0; for (i=2;i<=NF;i++) sum+=\$i; print sum; exit}' /proc/stat" \
    | awk 'NF {print $1; exit}'
}

proc_jiffies() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    printf ''
    return 0
  fi
  adb_shell "awk '{print \$14 + \$15}' /proc/$pid/stat 2>/dev/null" | awk 'NF {print $1; exit}'
}

cpu_usage_between() {
  local before_proc="$1" before_total="$2" after_proc="$3" after_total="$4"
  local cores
  cores="$(cpu_count)"
  if [[ -z "$before_proc" || -z "$before_total" || -z "$after_proc" || -z "$after_total" || -z "$cores" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v bp="$before_proc" -v bt="$before_total" -v ap="$after_proc" -v at="$after_total" -v cores="$cores" '
    BEGIN {
      dp = ap - bp;
      dt = at - bt;
      if (dt <= 0 || dp < 0) print "未采集";
      else printf "%.1f%%", dp / dt * cores * 100;
    }'
}

package_uid() {
  if [[ -z "$ANDROID_PACKAGE" ]]; then
    printf ''
    return 0
  fi
  adb_shell "cmd package list packages -U '$ANDROID_PACKAGE' 2>/dev/null | awk -F'uid:' 'NF > 1 {print \$2; exit}'" \
    | awk 'NF {print $1; exit}'
}

process_uid() {
  if [[ -n "$ANDROID_GPU_UID" ]]; then
    printf '%s' "$ANDROID_GPU_UID"
    return 0
  fi
  local uid user
  uid="$(package_uid)"
  if [[ -n "$uid" ]]; then
    printf '%s' "$uid"
    return 0
  fi
  if [[ -z "$ANDROID_PROCESS_NAME" ]]; then
    printf ''
    return 0
  fi
  user="$(adb_shell "ps -A -o USER,NAME | awk '\$2 == \"$ANDROID_PROCESS_NAME\" {print \$1; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "$user" ]]; then
    printf ''
    return 0
  fi
  adb_shell "id -u '$user' 2>/dev/null" | awk 'NF {print $1; exit}'
}

gpu_work_snapshot() {
  local uid="$1"
  if [[ -z "$uid" ]]; then
    printf ''
    return 0
  fi
  adb_shell "dumpsys gpu | awk -v uid='$uid' '\$2 == uid {print \$3, \$4; exit}'" | awk 'NF >= 2 {print $1, $2; exit}'
}

gpu_usage_between() {
  local before="$1" after="$2"
  if [[ -z "$before" || -z "$after" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v before="$before" -v after="$after" '
    BEGIN {
      split(before, b, " ");
      split(after, a, " ");
      da = a[1] - b[1];
      di = a[2] - b[2];
      dt = da + di;
      if (dt <= 0 || da < 0 || di < 0) print "未采集";
      else printf "%.1f%%", da / dt * 100;
    }'
}

gpu_freq() {
  local value
  value="$(adb_shell "for f in /sys/devices/platform/*mali*/cur_freq /sys/class/devfreq/*gpu*/cur_freq /sys/class/devfreq/*mali*/cur_freq; do [ -r \"\$f\" ] && cat \"\$f\" && exit 0; done" | awk 'NF {print $1; exit}')"
  if [[ -z "$value" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v hz="$value" 'BEGIN { printf "%.0f MHz", hz / 1000 }'
}

battery_temp() {
  local raw
  raw="$(adb_shell "dumpsys battery | awk '/temperature:/ {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "${raw:-}" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v raw="$raw" 'BEGIN { printf "%.1f°C", raw / 10 }'
}

thermal_status() {
  local status g3d
  status="$(adb_shell "dumpsys thermalservice | awk -F': ' '/Thermal Status:/ {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  g3d="$(adb_shell "dumpsys thermalservice | awk -F'[=,]' '/mName=G3D/ {for (i=1;i<=NF;i++) if (\$i ~ /mValue/) {print \$(i+1); exit}}'" | awk 'NF {print $1; exit}')"
  if [[ -z "${status:-}" && -z "${g3d:-}" ]]; then
    printf '未采集'
  elif [[ -z "${g3d:-}" ]]; then
    printf 'status=%s' "$status"
  elif [[ -z "${status:-}" ]]; then
    printf 'G3D=%s°C' "$g3d"
  else
    printf 'status=%s; G3D=%s°C' "$status" "$g3d"
  fi
}

openai_payload() {
  local max_tokens="$1" stream="$2" prompt="$3"
  jq -n \
    --arg model "$MODEL" \
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
  local max_tokens="$1" stream="$2" prompt="$3"
  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    --argjson stream "$stream" \
    '{
      model: $model,
      prompt: $prompt,
      stream: $stream,
      options: {
        num_predict: $max_tokens,
        temperature: 0
      }
    }'
}

detect_api_style() {
  if [[ "$API_STYLE" != "auto" ]]; then
    printf '%s' "$API_STYLE"
    return 0
  fi
  if curl -fsS --max-time 2 "$BASE_URL/v1/models" >/dev/null 2>&1; then
    printf 'openai'
    return 0
  fi
  if curl -fsS --max-time 2 "$BASE_URL/api/tags" >/dev/null 2>&1; then
    printf 'ollama'
    return 0
  fi
  echo "Cannot detect API style on $BASE_URL. Start the Android server and set DEVICE_PORT/API_STYLE." >&2
  exit 1
}

run_request() {
  local api="$1" max_tokens="$2" stream="$3" prompt="$4"
  if [[ "$api" == "openai" ]]; then
    curl -fsS -N --max-time "$REQUEST_TIMEOUT" "$BASE_URL/v1/chat/completions" \
      -H "Authorization: Bearer $API_KEY" \
      -H 'Content-Type: application/json' \
      -d "$(openai_payload "$max_tokens" "$stream" "$prompt")"
  elif [[ "$api" == "ollama" ]]; then
    curl -fsS -N --max-time "$REQUEST_TIMEOUT" "$BASE_URL/api/generate" \
      -H 'Content-Type: application/json' \
      -d "$(ollama_payload "$max_tokens" "$stream" "$prompt")"
  else
    echo "Unsupported API_STYLE=$api" >&2
    exit 1
  fi
}

ttft_ms_once() {
  local api="$1" tmp
  tmp="$(mktemp)"
  local ttft_s
  if [[ "$api" == "openai" ]]; then
    ttft_s="$(
      curl -sS -N --max-time "$REQUEST_TIMEOUT" -o "$tmp" -w '%{time_starttransfer}' \
        "$BASE_URL/v1/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H 'Content-Type: application/json' \
        -d "$(openai_payload "$TTFT_TOKENS" true "$PROMPT")" 2>/dev/null || true
    )"
  else
    ttft_s="$(
      curl -sS -N --max-time "$REQUEST_TIMEOUT" -o "$tmp" -w '%{time_starttransfer}' \
        "$BASE_URL/api/generate" \
        -H 'Content-Type: application/json' \
        -d "$(ollama_payload "$TTFT_TOKENS" true "$PROMPT")" 2>/dev/null || true
    )"
  fi
  rm -f "$tmp"
  if [[ -z "$ttft_s" || "$ttft_s" == "0.000000" ]]; then
    printf '未采集'
  else
    awk -v s="$ttft_s" 'BEGIN { printf "%.1f", s * 1000 }'
  fi
}

ttft_ms_p50() {
  local api="$1"
  run_request "$api" "$TTFT_TOKENS" true "$PROMPT" >/dev/null 2>&1 || true
  local values=()
  local i value
  for ((i=1; i<=TTFT_REPEATS; i++)); do
    value="$(ttft_ms_once "$api")"
    if [[ "$value" != "未采集" ]]; then
      values+=("$value")
    fi
  done
  if [[ "${#values[@]}" -eq 0 ]]; then
    printf '未采集'
    return 0
  fi
  printf '%s\n' "${values[@]}" | sort -n | awk '{a[NR]=$1} END { mid=int((NR+1)/2); printf "%.1f", a[mid] }'
}

completion_tokens_from_response() {
  local api="$1" file="$2"
  if [[ "$api" == "openai" ]]; then
    jq -r '(.usage.completion_tokens // .usage.output_tokens // 0)' "$file"
  else
    jq -r '(.eval_count // 0)' "$file"
  fi
}

benchmark_once() {
  local api="$1" tmp before_mem after_mem wall output_tokens tok_s ttft avg proc_mem battery thermal mem_delta
  local pid gpu_uid before_proc before_total after_proc after_total cpu_usage before_gpu after_gpu gpu_usage gpu_freq_after
  tmp="$(mktemp)"

  run_request "$api" "$WARMUP_TOKENS" false "$PROMPT" >/dev/null
  ttft="$(ttft_ms_p50 "$api")"

  pid="$(process_pid)"
  gpu_uid="$(process_uid)"
  before_mem="$(device_used_mb)"
  before_proc="$(proc_jiffies "$pid")"
  before_total="$(cpu_total_jiffies)"
  before_gpu="$(gpu_work_snapshot "$gpu_uid")"
  local start end
  start="$(perl -MTime::HiRes=time -e 'printf "%.6f", time')"
  run_request "$api" "$MAX_TOKENS" false "$PROMPT" >"$tmp"
  end="$(perl -MTime::HiRes=time -e 'printf "%.6f", time')"
  after_proc="$(proc_jiffies "$pid")"
  after_total="$(cpu_total_jiffies)"
  after_gpu="$(gpu_work_snapshot "$gpu_uid")"
  after_mem="$(device_used_mb)"

  wall="$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", e - s }')"
  avg="$wall"
  output_tokens="$(completion_tokens_from_response "$api" "$tmp")"
  if [[ -z "$output_tokens" || "$output_tokens" == "null" || "$output_tokens" == "0" ]]; then
    output_tokens="$MAX_TOKENS"
  fi
  tok_s="$(awk -v tok="$output_tokens" -v wall="$wall" 'BEGIN { if (wall > 0) printf "%.1f", tok / wall; else print "0.0" }')"
  cpu_usage="$(cpu_usage_between "$before_proc" "$before_total" "$after_proc" "$after_total")"
  gpu_usage="$(gpu_usage_between "$before_gpu" "$after_gpu")"
  gpu_freq_after="$(gpu_freq)"
  proc_mem="$(process_memory)"
  battery="$(battery_temp)"
  thermal="$(thermal_status)"
  mem_delta="$(mb_delta_to_gb "$before_mem" "$after_mem")"

  append_csv_row "$SERVICE" "$api" "$MODEL" "$ttft" "$output_tokens" "$wall" "$tok_s" "$avg" "$cpu_usage" "$gpu_usage" "$gpu_freq_after" "$proc_mem" "$mem_delta" "$battery" "$thermal"
  rm -f "$tmp"
}

ensure_device
adb forward --remove "tcp:$HOST_PORT" >/dev/null 2>&1 || true
adb forward "tcp:$HOST_PORT" "tcp:$DEVICE_PORT" >/dev/null

api="$(detect_api_style)"

printf '%s\n' '服务,API,模型,TTFT(ms),output tokens,total/wall_s,tokens/s,avg_request_s,CPU(%),GPU active(%),GPU freq,进程内存,系统内存增量,电池温度,thermal' > "$OUT"
benchmark_once "$api"

printf '%s\n' "$OUT"
