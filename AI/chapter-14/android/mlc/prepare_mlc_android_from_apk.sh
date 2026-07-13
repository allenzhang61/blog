#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/android}"
MLC_SRC="${MLC_SRC:-$ROOT/src/mlc-llm}"
MLC_REPO="${MLC_REPO:-https://github.com/mlc-ai/mlc-llm.git}"
MLC_REF="${MLC_REF:-main}"
APK_URL="${APK_URL:-https://github.com/mlc-ai/binary-mlc-llm-libs/releases/download/Android-09262024/mlc-chat.apk}"
APK_PATH="${APK_PATH:-$ROOT/downloads/mlc/mlc-chat-Android-09262024.apk}"
DECOMPILED_DIR="${DECOMPILED_DIR:-$ROOT/build/mlc-apk-src}"
PATCH_FILE="${PATCH_FILE:-$ROOT/mlc/patches/0001-add-mlc-benchmark-activity.patch}"
JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
ANDROID_SDK="${ANDROID_SDK:-$HOME/Library/Android/sdk}"

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
require_cmd curl
require_cmd unzip
require_cmd jadx

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$ANDROID_SDK/platform-tools:$PATH"

if [[ ! -d "$MLC_SRC/.git" ]]; then
  mkdir -p "$(dirname "$MLC_SRC")"
  git clone --depth 1 --branch "$MLC_REF" "$MLC_REPO" "$MLC_SRC"
else
  git -C "$MLC_SRC" fetch --depth 1 origin "$MLC_REF"
  git -C "$MLC_SRC" checkout -q FETCH_HEAD
fi

PATCH_ABS="$(cd "$(dirname "$PATCH_FILE")" && pwd)/$(basename "$PATCH_FILE")"
if git -C "$MLC_SRC" apply --check "$PATCH_ABS" >/dev/null 2>&1; then
  git -C "$MLC_SRC" apply "$PATCH_ABS"
fi

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
  require_cmd adb
  apk="$MLC_SRC/android/MLCEngineExample/app/build/outputs/apk/debug/app-debug.apk"
  size="$(stat -f%z "$apk")"
  adb push "$apk" /data/local/tmp/mlc-bench.apk >/dev/null
  session="$(adb shell cmd package install-create -r -d --user 0 -S "$size" | tr -d '\r' | sed -n 's/.*\[\([0-9]*\)\].*/\1/p')"
  adb shell cmd package install-write -S "$size" "$session" base /data/local/tmp/mlc-bench.apk >/dev/null
  adb shell cmd package install-commit "$session"
fi

echo "$MLC_SRC/android/MLCEngineExample"
