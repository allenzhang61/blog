#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/android}"
MLC_SRC="${MLC_SRC:-$ROOT/src/mlc-llm}"
MLC_REPO="${MLC_REPO:-https://github.com/mlc-ai/mlc-llm.git}"
MLC_REF="${MLC_REF:-main}"
MLC_APP="${MLC_APP:-MLCEngineExample}"
PATCH_FILE="${PATCH_FILE:-$ROOT/mlc/patches/0001-add-mlc-benchmark-activity.patch}"
PATCH_ABS="$(cd "$(dirname "$PATCH_FILE")" && pwd)/$(basename "$PATCH_FILE")"
MLC_LLM_BIN="${MLC_LLM_BIN:-}"
JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

MLC_MODEL="${MLC_MODEL:-HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC}"
MLC_MODEL_ID="${MLC_MODEL_ID:-Qwen2.5-0.5B-Instruct-q4f16_1-MLC}"
MLC_ESTIMATED_VRAM_BYTES="${MLC_ESTIMATED_VRAM_BYTES:-1000000000}"
MLC_PREFILL_CHUNK_SIZE="${MLC_PREFILL_CHUNK_SIZE:-1024}"
MLC_BUNDLE_WEIGHT="${MLC_BUNDLE_WEIGHT:-false}"

RUN_MLC_PACKAGE="${RUN_MLC_PACKAGE:-1}"
BUILD_APK="${BUILD_APK:-1}"
INSTALL_APK="${INSTALL_APK:-1}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required" >&2
    exit 1
  fi
}

require_cmd git

if [[ ! -d "$MLC_SRC/.git" ]]; then
  mkdir -p "$(dirname "$MLC_SRC")"
  git clone --depth 1 --branch "$MLC_REF" "$MLC_REPO" "$MLC_SRC"
else
  git -C "$MLC_SRC" fetch --depth 1 origin "$MLC_REF"
  git -C "$MLC_SRC" checkout -q FETCH_HEAD
fi

if git -C "$MLC_SRC" apply --check "$PATCH_ABS" >/dev/null 2>&1; then
  git -C "$MLC_SRC" apply "$PATCH_ABS"
else
  if [[ ! -f "$MLC_SRC/android/$MLC_APP/app/src/main/java/ai/mlc/mlcengineexample/BenchmarkActivity.kt" ]]; then
    echo "Patch cannot be applied, and BenchmarkActivity.kt was not found." >&2
    exit 1
  fi
fi

APP_DIR="$MLC_SRC/android/$MLC_APP"
CONFIG="$APP_DIR/mlc-package-config.json"

cat > "$CONFIG" <<JSON
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
  (
    cd "$APP_DIR"
    "$MLC_LLM_BIN" package
  )
fi

if [[ "$BUILD_APK" == "1" ]]; then
  (
    cd "$APP_DIR"
    ./gradlew assembleDebug
  )
fi

if [[ "$INSTALL_APK" == "1" ]]; then
  require_cmd adb
  APK="$APP_DIR/app/build/outputs/apk/debug/app-debug.apk"
  if [[ ! -f "$APK" ]]; then
    echo "APK not found: $APK" >&2
    exit 1
  fi
  adb install -r "$APK"
fi

echo "$APP_DIR"
