#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/android}"
COMMAND="${1:-help}"
shift || true

usage() {
  cat <<'EOF'
Usage:
  AI/chapter-14/android/android_llm.sh bench-http
  AI/chapter-14/android/android_llm.sh build-llama-vulkan
  AI/chapter-14/android/android_llm.sh mlc all
  AI/chapter-14/android/android_llm.sh mlc prepare
  AI/chapter-14/android/android_llm.sh mlc push-model
  AI/chapter-14/android/android_llm.sh mlc bench
  AI/chapter-14/android/android_llm.sh mlc package-prepare
  AI/chapter-14/android/android_llm.sh mediapipe all
  AI/chapter-14/android/android_llm.sh mediapipe prepare
  AI/chapter-14/android/android_llm.sh mediapipe push-model
  AI/chapter-14/android/android_llm.sh mediapipe bench

Aliases:
  bench                 same as bench-http
  build                 same as build-llama-vulkan
  mlc-all               same as mlc all
  mlc-prepare           same as mlc prepare
  mlc-push-model        same as mlc push-model
  mlc-bench             same as mlc bench
  mlc-package-prepare   same as mlc package-prepare
  mediapipe-all         same as mediapipe all
  mediapipe-prepare     same as mediapipe prepare
  mediapipe-push-model  same as mediapipe push-model
  mediapipe-bench       same as mediapipe bench
EOF
}

run_bench_http() {
  bash -s -- "$@" <<'__ANDROID_BENCH_HTTP__'

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
__ANDROID_BENCH_HTTP__
}

run_build_vulkan() {
  bash -s -- "$@" <<'__ANDROID_BUILD_LLAMA_VULKAN__'

ROOT="${ROOT:-AI/chapter-14/android}"
SRC_DIR="${SRC_DIR:-$ROOT/src/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/llama-android-vulkan}"
INSTALL_DIR="${INSTALL_DIR:-$ROOT/dist/llama-android-vulkan}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-/opt/homebrew/share/android-ndk}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-28}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/homebrew/opt/spirv-headers;/opt/homebrew/opt/spirv-tools;/opt/homebrew/opt/shaderc}"
SPIRV_HEADERS_DIR="${SPIRV_HEADERS_DIR:-/opt/homebrew/opt/spirv-headers/share/cmake/SPIRV-Headers}"
VULKAN_HEADERS_INCLUDE="${VULKAN_HEADERS_INCLUDE:-/opt/homebrew/opt/vulkan-headers/include}"
SPIRV_HEADERS_INCLUDE="${SPIRV_HEADERS_INCLUDE:-/opt/homebrew/opt/spirv-headers/include}"
ANDROID_CXX_FLAGS_RELEASE="${ANDROID_CXX_FLAGS_RELEASE:--O1 -DNDEBUG}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

require_cmd git
require_cmd cmake
require_cmd ninja
require_cmd glslc

TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN_FILE" ]]; then
  echo "Android NDK toolchain not found: $TOOLCHAIN_FILE" >&2
  echo "Install it with: brew install android-ndk" >&2
  exit 1
fi

mkdir -p "$(dirname "$SRC_DIR")" "$(dirname "$BUILD_DIR")" "$INSTALL_DIR"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone --depth 1 --branch "$LLAMA_REF" "$LLAMA_REPO" "$SRC_DIR"
else
  git -C "$SRC_DIR" fetch --depth 1 origin "$LLAMA_REF"
  git -C "$SRC_DIR" checkout FETCH_HEAD
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="$ANDROID_CXX_FLAGS_RELEASE" \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DCMAKE_CXX_FLAGS="-I$VULKAN_HEADERS_INCLUDE -I$SPIRV_HEADERS_INCLUDE" \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DSPIRV-Headers_DIR="$SPIRV_HEADERS_DIR" \
  -DVulkan_GLSLC_EXECUTABLE="$(command -v glslc)"

cmake --build "$BUILD_DIR" --target llama-server llama-cli llama-bench -j "$JOBS"

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

find "$BUILD_DIR" -maxdepth 3 -type f \( \
  -name 'llama-server' -o \
  -name 'llama-cli' -o \
  -name 'llama-bench' -o \
  -name '*.so' \
\) -exec cp -f {} "$INSTALL_DIR/" \;

chmod +x "$INSTALL_DIR"/llama-* 2>/dev/null || true

echo "$INSTALL_DIR"
__ANDROID_BUILD_LLAMA_VULKAN__
}

run_mlc() {
  local mlc_command="${1:-all}"
  shift || true
  bash -s -- "$mlc_command" "$@" <<'__ANDROID_MLC__'
COMMAND="${1:-all}"

ROOT="${ROOT:-AI/chapter-14/android}"
MLC_SRC="${MLC_SRC:-$ROOT/src/mlc-llm}"
MLC_REPO="${MLC_REPO:-https://github.com/mlc-ai/mlc-llm.git}"
MLC_REF="${MLC_REF:-a2bcc5c86678b72a86b7aadc29b643a5ce63c747}"
MLC_APP="${MLC_APP:-MLCEngineExample}"
PATCH_FILE="${PATCH_FILE:-$ROOT/mlc/patches/0001-add-mlc-benchmark-activity.patch}"
APK_URL="${APK_URL:-https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android-09262024/mlc-chat.apk}"
APK_PATH="${APK_PATH:-$ROOT/downloads/mlc/mlc-chat-Android-09262024.apk}"
DECOMPILED_DIR="${DECOMPILED_DIR:-$ROOT/build/mlc-apk-src}"
JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
ANDROID_SDK="${ANDROID_SDK:-$HOME/Library/Android/sdk}"

PACKAGE="${PACKAGE:-ai.mlc.mlcengineexample}"
ACTIVITY="${ACTIVITY:-ai.mlc.mlcengineexample.BenchmarkActivity}"
MODEL_ID="${MODEL_ID:-Qwen2.5-1.5B-Instruct-q4f16_1-MLC}"
MODEL_REPO="${MODEL_REPO:-mlc-ai/$MODEL_ID}"
MODEL_LIB="${MODEL_LIB:-qwen2_q4f16_1_2e221f430380225c03990ad24c3d030e}"
MODEL_PATH="${MODEL_PATH:-/data/data/$PACKAGE/files/$MODEL_ID}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT/downloads/mlc-models/$MODEL_ID}"
DEVICE_DIR="${DEVICE_DIR:-/data/data/$PACKAGE/files/$MODEL_ID}"
HF_BIN="${HF_BIN:-$ROOT/build/mlc-venv15/bin/hf}"

RESULT_DEVICE="${RESULT_DEVICE:-/sdcard/Android/data/$PACKAGE/files/mlc-bench-result.json}"
OUT="${OUT:-$ROOT/results/pixel9-mlc-bench-$(date +%Y%m%d-%H%M%S).csv}"
RAW_DIR="${RAW_DIR:-$ROOT/results}"
PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_TOKENS="${WARMUP_TOKENS:-4}"
TIMEOUT_S="${TIMEOUT_S:-420}"

MLC_MODEL="${MLC_MODEL:-HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC}"
MLC_MODEL_ID="${MLC_MODEL_ID:-Qwen2.5-0.5B-Instruct-q4f16_1-MLC}"
MLC_ESTIMATED_VRAM_BYTES="${MLC_ESTIMATED_VRAM_BYTES:-1000000000}"
MLC_PREFILL_CHUNK_SIZE="${MLC_PREFILL_CHUNK_SIZE:-1024}"
MLC_BUNDLE_WEIGHT="${MLC_BUNDLE_WEIGHT:-false}"
MLC_LLM_BIN="${MLC_LLM_BIN:-}"

BUILD_APK="${BUILD_APK:-1}"
INSTALL_APK="${INSTALL_APK:-1}"
RUN_MLC_PACKAGE="${RUN_MLC_PACKAGE:-1}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$ANDROID_SDK/platform-tools:$PATH"

usage() {
  cat <<'EOF'
Usage:
  AI/chapter-14/android/android_llm.sh mlc all
  AI/chapter-14/android/android_llm.sh mlc prepare
  AI/chapter-14/android/android_llm.sh mlc push-model
  AI/chapter-14/android/android_llm.sh mlc bench
  AI/chapter-14/android/android_llm.sh mlc package-prepare

Default all = prepare + push-model + bench.

Common env:
  MODEL_ID, MODEL_LIB, MODEL_PATH, MAX_TOKENS, WARMUP_TOKENS, OUT
  MLC_REF defaults to a2bcc5c86678b72a86b7aadc29b643a5ce63c747.
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

adb_shell() {
  adb shell "$@" 2>/dev/null | tr -d '\r'
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

checkout_mlc() {
  require_cmd git
  if [[ ! -d "$MLC_SRC/.git" ]]; then
    mkdir -p "$(dirname "$MLC_SRC")"
    git init -q "$MLC_SRC"
    git -C "$MLC_SRC" remote add origin "$MLC_REPO"
    git -C "$MLC_SRC" fetch --depth 1 origin "$MLC_REF"
    git -C "$MLC_SRC" checkout -q FETCH_HEAD
  else
    git -C "$MLC_SRC" fetch --depth 1 origin "$MLC_REF"
    git -C "$MLC_SRC" checkout -q FETCH_HEAD
  fi
}

apply_benchmark_patch() {
  local patch_abs
  patch_abs="$(cd "$(dirname "$PATCH_FILE")" && pwd)/$(basename "$PATCH_FILE")"
  if git -C "$MLC_SRC" apply --check "$patch_abs" >/dev/null 2>&1; then
    git -C "$MLC_SRC" apply "$patch_abs"
  elif [[ ! -f "$MLC_SRC/android/$MLC_APP/app/src/main/java/ai/mlc/mlcengineexample/BenchmarkActivity.kt" ]]; then
    echo "Patch cannot be applied, and BenchmarkActivity.kt was not found." >&2
    exit 1
  fi
}

install_apk_with_session() {
  require_cmd adb
  local apk="$1"
  local size session
  size="$(stat -f%z "$apk")"
  adb push "$apk" /data/local/tmp/mlc-bench.apk >/dev/null
  session="$(adb shell cmd package install-create -r -d --user 0 -S "$size" | tr -d '\r' | sed -n 's/.*\[\([0-9]*\)\].*/\1/p')"
  adb shell cmd package install-write -S "$size" "$session" base /data/local/tmp/mlc-bench.apk >/dev/null
  adb shell cmd package install-commit "$session"
}

prepare_from_apk() {
  require_cmd curl
  require_cmd unzip
  require_cmd jadx
  checkout_mlc
  apply_benchmark_patch

  mkdir -p "$(dirname "$APK_PATH")"
  if [[ ! -f "$APK_PATH" ]]; then
    curl -L -o "$APK_PATH" "$APK_URL"
  fi

  rm -rf "$DECOMPILED_DIR"
  jadx -q -d "$DECOMPILED_DIR" "$APK_PATH" >/dev/null || true

  mkdir -p "$MLC_SRC/android/mlc4j/output/arm64-v8a"
  unzip -p "$APK_PATH" lib/arm64-v8a/libtvm4j_runtime_packed.so \
    > "$MLC_SRC/android/mlc4j/output/arm64-v8a/libtvm4j_runtime_packed.so"

  rm -rf "$MLC_SRC/android/mlc4j/src/main/java/org/apache/tvm"
  mkdir -p "$MLC_SRC/android/mlc4j/src/main/java/org/apache"
  cp -R "$DECOMPILED_DIR/sources/org/apache/tvm" "$MLC_SRC/android/mlc4j/src/main/java/org/apache/"
  rm -rf \
    "$MLC_SRC/android/mlc4j/src/main/java/org/apache/tvm/rpc" \
    "$MLC_SRC/android/mlc4j/src/main/java/org/apache/tvm/contrib"
  cp "$MLC_SRC/3rdparty/tvm/jvm/core/src/main/java/org/apache/tvm/NativeLibraryLoader.java" \
    "$MLC_SRC/android/mlc4j/src/main/java/org/apache/tvm/NativeLibraryLoader.java"

  perl -0pi -e "s|project\\(':mlc4j'\\)\\.projectDir = file\\('dist/lib/mlc4j'\\)|project(':mlc4j').projectDir = file('../mlc4j')|" \
    "$MLC_SRC/android/MLCEngineExample/settings.gradle"
  perl -0pi -e 's/compileSdk 34/compileSdk 36/g' \
    "$MLC_SRC/android/MLCEngineExample/app/build.gradle" \
    "$MLC_SRC/android/mlc4j/build.gradle"
  printf 'sdk.dir=%s\n' "$ANDROID_SDK" > "$MLC_SRC/android/MLCEngineExample/local.properties"

  if [[ "$BUILD_APK" == "1" ]]; then
    (
      cd "$MLC_SRC/android/MLCEngineExample"
      ./gradlew assembleDebug
    )
  fi

  if [[ "$INSTALL_APK" == "1" ]]; then
    install_apk_with_session "$MLC_SRC/android/MLCEngineExample/app/build/outputs/apk/debug/app-debug.apk"
  fi

  echo "$MLC_SRC/android/MLCEngineExample"
}

prepare_with_mlc_package() {
  checkout_mlc
  apply_benchmark_patch

  local app_dir="$MLC_SRC/android/$MLC_APP"
  cat > "$app_dir/mlc-package-config.json" <<JSON
{
  "device": "android",
  "model_list": [
    {
      "model": "$MLC_MODEL",
      "estimated_vram_bytes": $MLC_ESTIMATED_VRAM_BYTES,
      "model_id": "$MLC_MODEL_ID",
      "bundle_weight": $MLC_BUNDLE_WEIGHT,
      "overrides": {
        "prefill_chunk_size": $MLC_PREFILL_CHUNK_SIZE
      }
    }
  ]
}
JSON

  if [[ "$RUN_MLC_PACKAGE" == "1" ]]; then
    if [[ -z "$MLC_LLM_BIN" ]]; then
      MLC_LLM_BIN="mlc_llm"
      require_cmd "$MLC_LLM_BIN"
    elif [[ ! -x "$MLC_LLM_BIN" ]]; then
      echo "MLC_LLM_BIN is not executable: $MLC_LLM_BIN" >&2
      exit 1
    fi
    (cd "$app_dir" && "$MLC_LLM_BIN" package)
  fi

  if [[ "$BUILD_APK" == "1" ]]; then
    (cd "$app_dir" && ./gradlew assembleDebug)
  fi

  if [[ "$INSTALL_APK" == "1" ]]; then
    require_cmd adb
    adb install -r "$app_dir/app/build/outputs/apk/debug/app-debug.apk"
  fi

  echo "$app_dir"
}

push_model() {
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
}

package_uid() {
  adb_shell "cmd package list packages -U '$PACKAGE' 2>/dev/null | awk -F'uid:' 'NF > 1 {print \$2; exit}'" \
    | awk 'NF {print $1; exit}'
}

package_pid() {
  adb_shell "pidof '$PACKAGE' 2>/dev/null || ps -A -o PID,NAME | awk '\$2 == \"$PACKAGE\" {print \$1; exit}'" \
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
  [[ -z "$pid" ]] && return 0
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

gpu_work_snapshot() {
  local uid="$1"
  [[ -z "$uid" ]] && return 0
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
      active = a[1] - b[1];
      inactive = a[2] - b[2];
      total = active + inactive;
      if (total <= 0 || active < 0 || inactive < 0) print "未采集";
      else printf "%.1f%%", active / total * 100;
    }'
}

gpu_freq() {
  local path hz
  path="$(adb_shell "find /sys/class/kgsl /sys/devices/platform /sys/class/devfreq -name cur_freq 2>/dev/null | grep -Ei 'mali|gpu|g3d|kgsl' | head -n 1" | awk 'NF {print $1; exit}')"
  if [[ -z "$path" ]]; then
    printf '未采集'
    return 0
  fi
  hz="$(adb_shell "cat '$path' 2>/dev/null" | awk 'NF {print $1; exit}')"
  if [[ -z "$hz" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v hz="$hz" 'BEGIN {
    if (hz >= 100000000) printf "%.0f MHz", hz / 1000000;
    else if (hz >= 100000) printf "%.0f MHz", hz / 1000;
    else printf "%s", hz;
  }'
}

device_used_mb() {
  local total avail
  total="$(adb_shell "awk '/MemTotal:/ {print \$2}' /proc/meminfo" | awk 'NF {print $1; exit}')"
  avail="$(adb_shell "awk '/MemAvailable:/ {print \$2}' /proc/meminfo" | awk 'NF {print $1; exit}')"
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
  local kb
  kb="$(adb_shell "dumpsys meminfo '$PACKAGE' | awk '/TOTAL PSS:/ {print \$3; exit} /^ *TOTAL / {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "${kb:-}" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v kb="$kb" 'BEGIN { printf "%.1f MB", kb / 1024 }'
}

battery_temp() {
  local raw
  raw="$(adb_shell "dumpsys battery | awk -F': ' '/temperature/ {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "$raw" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v raw="$raw" 'BEGIN { printf "%.1f°C", raw / 10 }'
}

thermal_summary() {
  local status g3d
  status="$(adb_shell "dumpsys thermalservice | awk -F': ' '/Thermal Status/ {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  g3d="$(adb_shell "dumpsys thermalservice | awk '/G3D|GPU|gpu/ {print; exit}'" | sed 's/^[[:space:]]*//')"
  if [[ -z "$status" && -z "$g3d" ]]; then
    printf '未采集'
  elif [[ -z "$g3d" ]]; then
    printf 'status=%s' "$status"
  else
    printf 'status=%s; %s' "${status:-未知}" "$g3d"
  fi
}

append_csv_header() {
  if [[ ! -f "$OUT" ]]; then
    printf '服务,API,模型,TTFT(ms),output tokens,total/wall_s,tokens/s,avg_request_s,CPU(%%),GPU active(%%),GPU freq,进程内存,系统内存增量,电池温度,thermal\n' > "$OUT"
  fi
}

append_csv_row() {
  local service="$1" api="$2" model="$3" ttft="$4" output_tokens="$5" wall="$6" tok_s="$7" avg="$8" cpu="$9" gpu="${10}" freq="${11}" proc_mem="${12}" mem_delta="${13}" battery="${14}" thermal="${15}"
  {
    csv_escape "$service"; printf ','
    csv_escape "$api"; printf ','
    csv_escape "$model"; printf ','
    printf '%s,%s,%s,%s,%s,' "$ttft" "$output_tokens" "$wall" "$tok_s" "$avg"
    csv_escape "$cpu"; printf ','
    csv_escape "$gpu"; printf ','
    csv_escape "$freq"; printf ','
    csv_escape "$proc_mem"; printf ','
    csv_escape "$mem_delta"; printf ','
    csv_escape "$battery"; printf ','
    csv_escape "$thermal"
    printf '\n'
  } >> "$OUT"
}

bench() {
  require_cmd adb
  require_cmd jq
  require_cmd awk
  mkdir -p "$(dirname "$OUT")" "$RAW_DIR"

  local device_count
  device_count="$(adb devices | awk 'NR > 1 && $2 == "device" { n++ } END { print n + 0 }')"
  if [[ "$device_count" -eq 0 ]]; then
    echo "No authorized Android device found." >&2
    adb devices -l >&2 || true
    exit 1
  fi

  if [[ -z "$MODEL_LIB" ]]; then
    echo "MODEL_LIB is required." >&2
    exit 1
  fi

  adb shell "rm -f '$RESULT_DEVICE'" >/dev/null
  adb shell am force-stop "$PACKAGE" >/dev/null 2>&1 || true

  local before_used uid gpu_before pid deadline cpu_before_proc cpu_before_total raw cpu_after_proc cpu_after_total gpu_after after_used status
  before_used="$(device_used_mb)"
  uid="$(package_uid)"
  gpu_before="$(gpu_work_snapshot "$uid")"

  adb shell \
    am start -W \
    -n "$PACKAGE/$ACTIVITY" \
    --es model_id "$MODEL_ID" \
    --es model_path "$MODEL_PATH" \
    --es model_lib "$MODEL_LIB" \
    --es prompt "$PROMPT" \
    --ei max_tokens "$MAX_TOKENS" \
    --ei warmup_tokens "$WARMUP_TOKENS" \
    --es result_path "$RESULT_DEVICE" >/dev/null

  pid=""
  deadline=$((SECONDS + 20))
  while [[ -z "$pid" && "$SECONDS" -lt "$deadline" ]]; do
    pid="$(package_pid)"
    sleep 0.2
  done

  cpu_before_proc="$(proc_jiffies "$pid")"
  cpu_before_total="$(cpu_total_jiffies)"

  deadline=$((SECONDS + TIMEOUT_S))
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    if adb shell "test -f '$RESULT_DEVICE'" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! adb shell "test -f '$RESULT_DEVICE'" >/dev/null 2>&1; then
    echo "Timeout waiting for $RESULT_DEVICE" >&2
    adb logcat -d -t 200 | grep -E 'MLCBench|AndroidRuntime|mlc' >&2 || true
    exit 1
  fi

  raw="$RAW_DIR/mlc-bench-$(date +%Y%m%d-%H%M%S).json"
  adb pull "$RESULT_DEVICE" "$raw" >/dev/null

  cpu_after_proc="$(proc_jiffies "$pid")"
  cpu_after_total="$(cpu_total_jiffies)"
  gpu_after="$(gpu_work_snapshot "$uid")"
  after_used="$(device_used_mb)"

  status="$(jq -r '.status // "unknown"' "$raw")"
  if [[ "$status" != "success" ]]; then
    cat "$raw" >&2
    exit 1
  fi

  append_csv_header
  append_csv_row \
    "$(jq -r '.service' "$raw")" \
    "$(jq -r '.api' "$raw")" \
    "$(jq -r '.model' "$raw")" \
    "$(jq -r '.ttft_ms' "$raw")" \
    "$(jq -r '.output_tokens' "$raw")" \
    "$(jq -r '.wall_s' "$raw")" \
    "$(jq -r '.tokens_s' "$raw")" \
    "$(jq -r '.avg_request_s' "$raw")" \
    "$(cpu_usage_between "$cpu_before_proc" "$cpu_before_total" "$cpu_after_proc" "$cpu_after_total")" \
    "$(gpu_usage_between "$gpu_before" "$gpu_after")" \
    "$(gpu_freq)" \
    "$(package_pss)" \
    "$(mb_delta_to_gb "$before_used" "$after_used")" \
    "$(battery_temp)" \
    "$(thermal_summary)"

  adb shell am force-stop "$PACKAGE" >/dev/null 2>&1 || true
  echo "$OUT"
}

case "$COMMAND" in
  all)
    prepare_from_apk
    push_model
    bench
    ;;
  prepare)
    prepare_from_apk
    ;;
  package-prepare)
    prepare_with_mlc_package
    ;;
  push-model)
    push_model
    ;;
  bench)
    bench
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
__ANDROID_MLC__
}

run_mediapipe() {
  local mediapipe_command="${1:-all}"
  shift || true
  bash -s -- "$mediapipe_command" "$@" <<'__ANDROID_MEDIAPIPE__'
set -euo pipefail

COMMAND="${1:-all}"

ROOT="${ROOT:-AI/chapter-14/android}"
MP_SRC="${MP_SRC:-$ROOT/src/mediapipe-samples}"
MP_APP_DIR="${MP_APP_DIR:-$MP_SRC/examples/llm_inference/android}"
MP_PACKAGE="${MP_PACKAGE:-com.google.mediapipe.examples.llminference}"
MP_ACTIVITY="${MP_ACTIVITY:-com.google.mediapipe.examples.llminference.BenchmarkActivity}"
ANDROID_SDK="${ANDROID_SDK:-$HOME/Library/Android/sdk}"
JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"

MP_MODEL_NAME="${MP_MODEL_NAME:-SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task}"
MP_MODEL_URL="${MP_MODEL_URL:-https://huggingface.co/litert-community/SmolLM-135M-Instruct/resolve/main/SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task}"
MP_MODEL_PATH="${MP_MODEL_PATH:-/data/local/tmp/$MP_MODEL_NAME}"
MP_LOCAL_MODEL="${MP_LOCAL_MODEL:-$ROOT/downloads/mediapipe/$MP_MODEL_NAME}"
MP_BACKEND="${MP_BACKEND:-CPU}"
MP_RESULT_DEVICE="${MP_RESULT_DEVICE:-/sdcard/Android/data/$MP_PACKAGE/files/mediapipe-bench-result.json}"
OUT="${OUT:-$ROOT/results/pixel9-mediapipe-bench-$(date +%Y%m%d-%H%M%S).csv}"
RAW_DIR="${RAW_DIR:-$ROOT/results}"
PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_TOKENS="${WARMUP_TOKENS:-4}"
TIMEOUT_S="${TIMEOUT_S:-300}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$ANDROID_SDK/platform-tools:$PATH"

usage() {
  cat <<'EOF'
Usage:
  AI/chapter-14/android/android_llm.sh mediapipe all
  AI/chapter-14/android/android_llm.sh mediapipe prepare
  AI/chapter-14/android/android_llm.sh mediapipe push-model
  AI/chapter-14/android/android_llm.sh mediapipe bench

Default all = prepare + push-model + bench.

Common env:
  MP_MODEL_URL, MP_MODEL_NAME, MP_MODEL_PATH, MP_BACKEND, MAX_TOKENS, WARMUP_TOKENS, OUT
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

adb_shell() {
  adb shell "$@" 2>/dev/null | tr -d '\r'
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_csv_header() {
  if [[ ! -f "$OUT" ]]; then
    printf '服务,API,模型,TTFT(ms),output tokens,total/wall_s,tokens/s,avg_request_s,CPU(%%),GPU active(%%),GPU freq,进程内存,系统内存增量,电池温度,thermal\n' > "$OUT"
  fi
}

append_csv_row() {
  local service="$1" api="$2" model="$3" ttft="$4" output_tokens="$5" wall="$6" tok_s="$7" avg="$8" cpu="$9" gpu="${10}" freq="${11}" proc_mem="${12}" mem_delta="${13}" battery="${14}" thermal="${15}"
  {
    csv_escape "$service"; printf ','
    csv_escape "$api"; printf ','
    csv_escape "$model"; printf ','
    printf '%s,%s,%s,%s,%s,' "$ttft" "$output_tokens" "$wall" "$tok_s" "$avg"
    csv_escape "$cpu"; printf ','
    csv_escape "$gpu"; printf ','
    csv_escape "$freq"; printf ','
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
    echo "No authorized Android device found." >&2
    adb devices -l >&2 || true
    exit 1
  fi
}

package_uid() {
  adb_shell "cmd package list packages -U '$MP_PACKAGE' 2>/dev/null | awk -F'uid:' 'NF > 1 {print \$2; exit}'" \
    | awk 'NF {print $1; exit}'
}

package_pid() {
  adb_shell "pidof '$MP_PACKAGE' 2>/dev/null || ps -A -o PID,NAME | awk '\$2 == \"$MP_PACKAGE\" {print \$1; exit}'" \
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
  [[ -z "$pid" ]] && return 0
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

gpu_work_snapshot() {
  local uid="$1"
  [[ -z "$uid" ]] && return 0
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
      active = a[1] - b[1];
      inactive = a[2] - b[2];
      total = active + inactive;
      if (total <= 0 || active < 0 || inactive < 0) print "未采集";
      else printf "%.1f%%", active / total * 100;
    }'
}

gpu_freq() {
  local path hz
  path="$(adb_shell "find /sys/class/kgsl /sys/devices/platform /sys/class/devfreq -name cur_freq 2>/dev/null | grep -Ei 'mali|gpu|g3d|kgsl' | head -n 1" | awk 'NF {print $1; exit}')"
  if [[ -z "$path" ]]; then
    printf '未采集'
    return 0
  fi
  hz="$(adb_shell "cat '$path' 2>/dev/null" | awk 'NF {print $1; exit}')"
  if [[ -z "$hz" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v hz="$hz" 'BEGIN {
    if (hz >= 100000000) printf "%.0f MHz", hz / 1000000;
    else if (hz >= 100000) printf "%.0f MHz", hz / 1000;
    else printf "%s", hz;
  }'
}

device_used_mb() {
  local total avail
  total="$(adb_shell "awk '/MemTotal:/ {print \$2}' /proc/meminfo" | awk 'NF {print $1; exit}')"
  avail="$(adb_shell "awk '/MemAvailable:/ {print \$2}' /proc/meminfo" | awk 'NF {print $1; exit}')"
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
  local kb
  kb="$(adb_shell "dumpsys meminfo '$MP_PACKAGE' | awk '/TOTAL PSS:/ {print \$3; exit} /^ *TOTAL / {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "${kb:-}" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v kb="$kb" 'BEGIN { printf "%.1f MB", kb / 1024 }'
}

battery_temp() {
  local raw
  raw="$(adb_shell "dumpsys battery | awk -F': ' '/temperature/ {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  if [[ -z "$raw" ]]; then
    printf '未采集'
    return 0
  fi
  awk -v raw="$raw" 'BEGIN { printf "%.1f°C", raw / 10 }'
}

thermal_summary() {
  local status g3d
  status="$(adb_shell "dumpsys thermalservice | awk -F': ' '/Thermal Status/ {print \$2; exit}'" | awk 'NF {print $1; exit}')"
  g3d="$(adb_shell "dumpsys thermalservice | awk '/G3D|GPU|gpu/ {print; exit}'" | sed 's/^[[:space:]]*//')"
  if [[ -z "$status" && -z "$g3d" ]]; then
    printf '未采集'
  elif [[ -z "$g3d" ]]; then
    printf 'status=%s' "$status"
  else
    printf 'status=%s; %s' "${status:-未知}" "$g3d"
  fi
}

write_benchmark_activity() {
  local src_dir manifest
  src_dir="$MP_APP_DIR/app/src/main/java/com/google/mediapipe/examples/llminference"
  manifest="$MP_APP_DIR/app/src/main/AndroidManifest.xml"
  mkdir -p "$src_dir"
  cat > "$src_dir/BenchmarkActivity.kt" <<'KOTLIN'
package com.google.mediapipe.examples.llminference

import android.app.Activity
import android.os.Bundle
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.Backend
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class BenchmarkActivity : Activity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    Thread { runBenchmark() }.start()
  }

  private fun runBenchmark() {
    val modelPath = intent.getStringExtra("model_path") ?: "/data/local/tmp/SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task"
    val modelName = intent.getStringExtra("model_name") ?: File(modelPath).name
    val backendName = intent.getStringExtra("backend") ?: "CPU"
    val prompt = intent.getStringExtra("prompt") ?: "请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。"
    val maxTokens = intent.getIntExtra("max_tokens", 128)
    val warmupTokens = intent.getIntExtra("warmup_tokens", 4)
    val resultPath = intent.getStringExtra("result_path") ?: "${getExternalFilesDir(null)?.absolutePath}/mediapipe-bench-result.json"
    val backend = if (backendName.equals("GPU", ignoreCase = true)) Backend.GPU else Backend.CPU
    val startedNs = System.nanoTime()
    var inference: LlmInference? = null
    var session: LlmInferenceSession? = null
    try {
      val inferenceOptions = LlmInference.LlmInferenceOptions.builder()
        .setModelPath(modelPath)
        .setMaxTokens(maxOf(maxTokens + 256, 512))
        .setPreferredBackend(backend)
        .build()
      inference = LlmInference.createFromOptions(this, inferenceOptions)
      val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
        .setTemperature(0.0f)
        .setTopK(1)
        .setTopP(1.0f)
        .build()
      session = LlmInferenceSession.createFromOptions(inference, sessionOptions)

      runGenerate(session, "Say OK.", warmupTokens)
      session.close()
      session = LlmInferenceSession.createFromOptions(inference, sessionOptions)

      val result = runGenerate(session, prompt, maxTokens)
      val wallS = result.wallMs / 1000.0
      val tokensS = if (wallS > 0.0) result.outputTokens / wallS else 0.0
      val json = """
        {
          "status":"success",
          "service":"Google AI Edge Gallery / MediaPipe LLM Inference",
          "api":"mediapipe-llm-inference",
          "model":"$modelName ($backendName)",
          "ttft_ms":${"%.1f".format(result.ttftMs)},
          "output_tokens":${result.outputTokens},
          "wall_s":${"%.2f".format(wallS)},
          "tokens_s":${"%.1f".format(tokensS)},
          "avg_request_s":${"%.2f".format(wallS)},
          "response_chars":${result.text.length},
          "load_s":${"%.2f".format((result.startNs - startedNs) / 1_000_000_000.0)}
        }
      """.trimIndent()
      writeResult(resultPath, json)
    } catch (t: Throwable) {
      writeResult(resultPath, "{\"status\":\"error\",\"error\":${quote(t.stackTraceToString())}}")
    } finally {
      try { session?.close() } catch (_: Throwable) {}
      try { inference?.close() } catch (_: Throwable) {}
      finish()
    }
  }

  private data class GenerateResult(
    val text: String,
    val outputTokens: Int,
    val ttftMs: Double,
    val wallMs: Double,
    val startNs: Long,
  )

  private fun runGenerate(session: LlmInferenceSession, prompt: String, maxTokens: Int): GenerateResult {
    val firstTokenSeen = AtomicBoolean(false)
    val firstNs = AtomicReference<Long>(0L)
    val text = StringBuilder()
    val startNs = System.nanoTime()
    session.addQueryChunk(prompt)
    val future = session.generateResponseAsync { partial, done ->
      if (!firstTokenSeen.getAndSet(true)) {
        firstNs.set(System.nanoTime())
      }
      text.append(partial)
    }
    val finalText = future.get(240, TimeUnit.SECONDS)
    val endNs = System.nanoTime()
    val fullText = finalText.ifBlank { text.toString() }
    val outputTokens = estimateOutputTokens(fullText, maxTokens)
    val ttftMs = if (firstNs.get() > 0L) (firstNs.get() - startNs) / 1_000_000.0 else -1.0
    return GenerateResult(fullText, outputTokens, ttftMs, (endNs - startNs) / 1_000_000.0, startNs)
  }

  private fun estimateOutputTokens(text: String, maxTokens: Int): Int {
    if (text.isBlank()) return 0
    val rough = Regex("""[\p{L}\p{N}]+|[^\s]""").findAll(text).count()
    return rough.coerceAtMost(maxTokens)
  }

  private fun writeResult(path: String, content: String) {
    val file = File(path)
    file.parentFile?.mkdirs()
    file.writeText(content)
  }

  private fun quote(value: String): String {
    return "\"" + value
      .replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")
      .replace("\t", "\\t") + "\""
  }
}
KOTLIN

  if ! grep -q 'BenchmarkActivity' "$manifest"; then
    perl -0pi -e 's#(<activity\s+android:name="com\.google\.mediapipe\.examples\.llminference\.MainActivity")#<activity android:name="com.google.mediapipe.examples.llminference.BenchmarkActivity" android:exported="true" />\n\n        $1#' "$manifest"
  fi
}

install_apk_with_session() {
  require_cmd adb
  local apk="$1"
  local size session
  size="$(stat -f%z "$apk")"
  adb push "$apk" /data/local/tmp/mediapipe-bench.apk >/dev/null
  session="$(adb shell cmd package install-create -r -d --user 0 -S "$size" | tr -d '\r' | sed -n 's/.*\[\([0-9]*\)\].*/\1/p')"
  adb shell cmd package install-write -S "$size" "$session" base /data/local/tmp/mediapipe-bench.apk >/dev/null
  adb shell cmd package install-commit "$session"
}

prepare() {
  require_cmd git
  require_cmd adb
  if [[ ! -d "$MP_SRC/.git" ]]; then
    mkdir -p "$(dirname "$MP_SRC")"
    git clone --depth 1 https://github.com/google-ai-edge/mediapipe-samples.git "$MP_SRC"
  fi
  write_benchmark_activity
  printf 'sdk.dir=%s\n' "$ANDROID_SDK" > "$MP_APP_DIR/local.properties"
  (cd "$MP_APP_DIR" && ./gradlew assembleDebug)
  install_apk_with_session "$MP_APP_DIR/app/build/outputs/apk/debug/app-debug.apk"
}

push_model() {
  require_cmd curl
  require_cmd adb
  mkdir -p "$(dirname "$MP_LOCAL_MODEL")"
  if [[ ! -f "$MP_LOCAL_MODEL" ]]; then
    curl -L --fail --continue-at - -o "$MP_LOCAL_MODEL" "$MP_MODEL_URL"
  fi
  adb push "$MP_LOCAL_MODEL" "$MP_MODEL_PATH"
}

bench() {
  require_cmd adb
  require_cmd jq
  mkdir -p "$(dirname "$OUT")" "$RAW_DIR"
  ensure_device
  adb shell "rm -f '$MP_RESULT_DEVICE'" >/dev/null 2>&1 || true
  adb shell am force-stop "$MP_PACKAGE" >/dev/null 2>&1 || true

  local before_used uid gpu_before pid deadline cpu_before_proc cpu_before_total raw cpu_after_proc cpu_after_total gpu_after after_used status
  before_used="$(device_used_mb)"
  uid="$(package_uid)"
  gpu_before="$(gpu_work_snapshot "$uid")"

  adb shell \
    am start -W \
    -n "$MP_PACKAGE/$MP_ACTIVITY" \
    --es model_path "$MP_MODEL_PATH" \
    --es model_name "$MP_MODEL_NAME" \
    --es backend "$MP_BACKEND" \
    --es prompt "$PROMPT" \
    --ei max_tokens "$MAX_TOKENS" \
    --ei warmup_tokens "$WARMUP_TOKENS" \
    --es result_path "$MP_RESULT_DEVICE" >/dev/null

  pid=""
  deadline=$((SECONDS + 20))
  while [[ -z "$pid" && "$SECONDS" -lt "$deadline" ]]; do
    pid="$(package_pid)"
    sleep 0.2
  done

  cpu_before_proc="$(proc_jiffies "$pid")"
  cpu_before_total="$(cpu_total_jiffies)"

  deadline=$((SECONDS + TIMEOUT_S))
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    if adb shell "test -f '$MP_RESULT_DEVICE'" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! adb shell "test -f '$MP_RESULT_DEVICE'" >/dev/null 2>&1; then
    echo "Timeout waiting for $MP_RESULT_DEVICE" >&2
    adb logcat -d -t 300 | grep -E 'BenchmarkActivity|llminference|mediapipe|AndroidRuntime' >&2 || true
    exit 1
  fi

  raw="$RAW_DIR/mediapipe-bench-$(date +%Y%m%d-%H%M%S).json"
  adb pull "$MP_RESULT_DEVICE" "$raw" >/dev/null

  cpu_after_proc="$(proc_jiffies "$pid")"
  cpu_after_total="$(cpu_total_jiffies)"
  gpu_after="$(gpu_work_snapshot "$uid")"
  after_used="$(device_used_mb)"

  status="$(jq -r '.status // "unknown"' "$raw")"
  if [[ "$status" != "success" ]]; then
    cat "$raw" >&2
    exit 1
  fi

  append_csv_header
  append_csv_row \
    "$(jq -r '.service' "$raw")" \
    "$(jq -r '.api' "$raw")" \
    "$(jq -r '.model' "$raw")" \
    "$(jq -r '.ttft_ms' "$raw")" \
    "$(jq -r '.output_tokens' "$raw")" \
    "$(jq -r '.wall_s' "$raw")" \
    "$(jq -r '.tokens_s' "$raw")" \
    "$(jq -r '.avg_request_s' "$raw")" \
    "$(cpu_usage_between "$cpu_before_proc" "$cpu_before_total" "$cpu_after_proc" "$cpu_after_total")" \
    "$(gpu_usage_between "$gpu_before" "$gpu_after")" \
    "$(gpu_freq)" \
    "$(package_pss)" \
    "$(mb_delta_to_gb "$before_used" "$after_used")" \
    "$(battery_temp)" \
    "$(thermal_summary)"

  adb shell am force-stop "$MP_PACKAGE" >/dev/null 2>&1 || true
  echo "$OUT"
}

case "$COMMAND" in
  all)
    prepare
    push_model
    bench
    ;;
  prepare)
    prepare
    ;;
  push-model)
    push_model
    ;;
  bench)
    bench
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown mediapipe command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
__ANDROID_MEDIAPIPE__
}

case "$COMMAND" in
  bench|bench-http) run_bench_http "$@" ;;
  build|build-llama-vulkan) run_build_vulkan "$@" ;;
  mlc) run_mlc "$@" ;;
  mlc-all) run_mlc all "$@" ;;
  mlc-prepare) run_mlc prepare "$@" ;;
  mlc-push-model) run_mlc push-model "$@" ;;
  mlc-bench) run_mlc bench "$@" ;;
  mlc-package-prepare) run_mlc package-prepare "$@" ;;
  mediapipe) run_mediapipe "$@" ;;
  mediapipe-all) run_mediapipe all "$@" ;;
  mediapipe-prepare) run_mediapipe prepare "$@" ;;
  mediapipe-push-model) run_mediapipe push-model "$@" ;;
  mediapipe-bench) run_mediapipe bench "$@" ;;
  help|-h|--help) usage ;;
  *) echo "Unknown command: $COMMAND" >&2; usage >&2; exit 1 ;;
esac
