#!/usr/bin/env bash
set -euo pipefail

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
