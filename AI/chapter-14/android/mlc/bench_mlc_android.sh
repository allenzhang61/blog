#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/android}"
OUT="${OUT:-$ROOT/results/pixel9-mlc-bench-$(date +%Y%m%d-%H%M%S).csv}"
RAW_DIR="${RAW_DIR:-$ROOT/results}"

PACKAGE="${PACKAGE:-ai.mlc.mlcengineexample}"
ACTIVITY="${ACTIVITY:-ai.mlc.mlcengineexample.BenchmarkActivity}"
RESULT_DEVICE="${RESULT_DEVICE:-/sdcard/Android/data/$PACKAGE/files/mlc-bench-result.json}"

MODEL_ID="${MODEL_ID:-Qwen2.5-0.5B-Instruct-q4f16_1-MLC}"
MODEL_PATH="${MODEL_PATH:-/sdcard/Android/data/$PACKAGE/files/$MODEL_ID}"
MODEL_LIB="${MODEL_LIB:-}"
PROMPT="${PROMPT:-请用中文简洁说明大语言模型推理中 prefill 和 decode 的区别，控制在一百字以内。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_TOKENS="${WARMUP_TOKENS:-4}"
TIMEOUT_S="${TIMEOUT_S:-240}"

mkdir -p "$(dirname "$OUT")" "$RAW_DIR"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

require_cmd adb
require_cmd jq
require_cmd awk

adb_shell() {
  adb shell "$@" 2>/dev/null | tr -d '\r'
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
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
  local path
  path="$(adb_shell "find /sys/class/kgsl /sys/devices/platform /sys/class/devfreq -name cur_freq 2>/dev/null | grep -Ei 'mali|gpu|g3d|kgsl' | head -n 1" | awk 'NF {print $1; exit}')"
  if [[ -z "$path" ]]; then
    printf '未采集'
    return 0
  fi
  local hz
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

device_count="$(adb devices | awk 'NR > 1 && $2 == "device" { n++ } END { print n + 0 }')"
if [[ "$device_count" -eq 0 ]]; then
  echo "No authorized Android device found." >&2
  adb devices -l >&2 || true
  exit 1
fi

if [[ -z "$MODEL_LIB" ]]; then
  echo "MODEL_LIB is required. Use the model_lib generated by mlc_llm package." >&2
  exit 1
fi

adb shell "rm -f '$RESULT_DEVICE'" >/dev/null
adb shell am force-stop "$PACKAGE" >/dev/null 2>&1 || true

before_used="$(device_used_mb)"
uid="$(package_uid)"
gpu_before="$(gpu_work_snapshot "$uid")"

start_args=(
  am start -W
  -n "$PACKAGE/$ACTIVITY"
  --es model_id "$MODEL_ID"
  --es model_path "$MODEL_PATH"
  --es model_lib "$MODEL_LIB"
  --es prompt "$PROMPT"
  --ei max_tokens "$MAX_TOKENS"
  --ei warmup_tokens "$WARMUP_TOKENS"
  --es result_path "$RESULT_DEVICE"
)

adb shell "${start_args[@]}" >/dev/null

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

echo "$OUT"
