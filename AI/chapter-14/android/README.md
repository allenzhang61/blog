# Pixel 9 本地 LLM Server 压测脚本

这里放 Android / Pixel 9 上本地 LLM server 的压测脚本。脚本从 Mac 侧通过 `adb forward` 访问手机上的 HTTP 服务，不需要让手机和 Mac 在同一个局域网里。

## 前置条件

Mac 侧需要：

```bash
brew install android-platform-tools jq
```

Pixel 9 侧需要：

- 打开开发者选项和 USB 调试。
- 手机上已经启动一个 LLM HTTP server。
- server 至少支持 OpenAI-compatible `/v1/chat/completions` 或 Ollama-compatible `/api/generate` 其中一种 API。

## 推荐测试对象

优先测试：

- ServLlama：Android App 形式包装 `llama.cpp` / `llama-server`。
- Termux + `llama.cpp`：自己编译或安装 `llama-server` 后启动 HTTP API。
- 其他 OpenAI-compatible Android LLM server：只要能通过手机本地端口暴露 `/v1/chat/completions` 即可。

MLC LLM、Google AI Edge Gallery、MediaPipe LLM Inference 更偏 App / SDK。它们本身不一定暴露 HTTP API；如果要用通用 HTTP 脚本压测，需要额外包一层 HTTP server。当前仓库已经为 MLC LLM Android 增加了源码 patch 路线，见 `AI/chapter-14/android/mlc/`。

## 运行

默认只做单请求压测，不做并发大于 1 的压力测试：

```bash
SERVICE="ServLlama" \
API_STYLE=openai \
DEVICE_PORT=8080 \
MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf" \
ANDROID_PACKAGE="io.github.arkanefans.servllama" \
  AI/chapter-14/android/android_llm.sh bench-http
```

如果是通过 adb 直接启动的二进制进程，可以用 `ANDROID_PROCESS_NAME` 采集进程 RSS：

```bash
SERVICE="llama.cpp llama-server" \
API_STYLE=openai \
DEVICE_PORT=8080 \
MODEL="Qwen3-0.6B-Q4_K_M.gguf" \
ANDROID_PROCESS_NAME="llama-server" \
  AI/chapter-14/android/android_llm.sh bench-http
```

如果手机端是 Ollama-compatible API：

```bash
SERVICE="Android Ollama-compatible" \
API_STYLE=ollama \
DEVICE_PORT=11434 \
MODEL="qwen2.5:1.5b" \
  AI/chapter-14/android/android_llm.sh bench-http
```

如果不确定 API 类型，可以用：

```bash
DEVICE_PORT=8080 API_STYLE=auto AI/chapter-14/android/android_llm.sh bench-http
```

如果是 MLC LLM Android，使用源码 patch 增加 `BenchmarkActivity`，再通过 adb 直接启动 App 内 benchmark：

```bash
AI/chapter-14/android/android_llm.sh mlc all
```

如果只重新压测：

```bash
MODEL_ID="Qwen2.5-1.5B-Instruct-q4f16_1-MLC" \
MODEL_LIB="qwen2_q4f16_1_2e221f430380225c03990ad24c3d030e" \
MODEL_PATH="/data/data/ai.mlc.mlcengineexample/files/Qwen2.5-1.5B-Instruct-q4f16_1-MLC" \
AI/chapter-14/android/android_llm.sh mlc bench
```

结果默认输出到：

```text
AI/chapter-14/android/results/pixel9-bench-YYYYMMDD-HHMMSS.csv
```

CSV 列与 `AI/chapter-14/14-013.md` 中的压测表保持一致。

MLC 路线的 CSV 也使用相同列；区别是 `API` 标记为 `mlc-engine`，TTFT 和 tokens/s 来自 App 内部直接调用 `MLCEngine` 的计时。

如果是 Google AI Edge Gallery / MediaPipe LLM Inference 路线，使用官方 `mediapipe-samples`，再额外插入 `BenchmarkActivity`：

```bash
AI/chapter-14/android/android_llm.sh mediapipe all
```

如果只重新压测：

```bash
MP_MODEL_NAME="Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task" \
MP_MODEL_URL="https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct/resolve/main/Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task" \
MP_MODEL_PATH="/data/local/tmp/Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task" \
AI/chapter-14/android/android_llm.sh mediapipe bench
```

MediaPipe 路线的 CSV 也使用相同列；区别是 `API` 标记为 `mediapipe-llm-inference`，TTFT 和 tokens/s 来自 App 内部直接调用 `LlmInferenceSession` 的计时。

## 构建 llama.cpp Android Vulkan

如果要验证 Pixel 9 GPU 路线，可以用下面的脚本交叉编译带 Vulkan backend 的 `llama.cpp`：

```bash
AI/chapter-14/android/android_llm.sh build-llama-vulkan
```

脚本会：

- 拉取 `ggml-org/llama.cpp` 源码。
- 使用 Android NDK 编译 `arm64-v8a`。
- 开启 `-DGGML_VULKAN=ON`。
- 输出到 `AI/chapter-14/android/dist/llama-android-vulkan/`。

需要的本机依赖：

```bash
brew install cmake ninja shaderc spirv-headers spirv-tools vulkan-headers android-ndk
```

推送并验证设备：

```bash
adb shell 'rm -rf /data/local/tmp/llama-vulkan && mkdir -p /data/local/tmp/llama-vulkan'
adb push AI/chapter-14/android/dist/llama-android-vulkan/. /data/local/tmp/llama-vulkan/
adb shell 'chmod -R 755 /data/local/tmp/llama-vulkan'
adb shell 'cd /data/local/tmp/llama-vulkan && LD_LIBRARY_PATH=/data/local/tmp/llama-vulkan ./llama-server --list-devices'
```

如果看到类似下面的输出，说明 Vulkan backend 可用：

```text
Available devices:
  Vulkan0: Mali-G715
```

## 指标口径

- `TTFT(ms)`：用 streaming 请求的 `curl time_starttransfer` 近似首块响应时间，先 warmup，再取多次 p50。
- `tokens/s`：单请求输出 token 数 / 请求总耗时。
- `CPU(%)`：用 `/proc/<pid>/stat` 和 `/proc/stat` 前后差值估算进程 CPU。Android 多核机器上可能超过 100%，例如 600% 约等于 6 个 CPU core。
- `GPU active(%)`：用 `dumpsys gpu` 中目标 UID 的 `total_active_duration_ns / total_inactive_duration_ns` 前后差值估算。adb 直接启动的二进制通常归属 `shell` UID，也就是 `2000`；这是 UID 级别口径，不是严格 PID 级别。
- `GPU freq`：读取 Mali `cur_freq`，只是采样点频率，不等同于平均频率。
- `进程内存`：如果设置 `ANDROID_PACKAGE`，从 `adb shell dumpsys meminfo <package>` 读取 App PSS；如果设置 `ANDROID_PROCESS_NAME`，从 `ps` 读取进程 RSS。
- `系统内存增量`：请求前后 Android `MemAvailable` 差值的粗略估计，只适合作为参考。
- `thermal`：读取 `dumpsys thermalservice` 中的 thermal status 和 G3D 温度。Android 非 root 环境通常不能稳定采集真实 GPU utilization。

## 常见问题

如果看到：

```text
Cannot detect API style on http://127.0.0.1:18080
```

说明手机端目标端口还没有 HTTP LLM server，或者 `DEVICE_PORT` 设置错了。可以先手工探测：

```bash
adb forward tcp:18080 tcp:8080
curl http://127.0.0.1:18080/v1/models
curl http://127.0.0.1:18080/api/tags
```
