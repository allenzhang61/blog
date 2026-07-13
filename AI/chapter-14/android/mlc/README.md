# MLC LLM Android adb 压测

这里放 MLC LLM Android 的源码 patch 和 adb 压测脚本。

MLC Android 默认是 App / SDK 形态，不像 `llama.cpp llama-server` 那样天然暴露 HTTP 端口。为了让压测口径稳定，这里采用“改源码加 benchmark Activity”的方式：

1. 在 `MLCEngineExample` 中新增 `BenchmarkActivity`。
2. Mac 侧通过 `adb shell am start` 传入模型、prompt、输出 token 数。
3. App 内部直接调用 `MLCEngine`，记录 TTFT、总耗时、输出 token、tokens/s。
4. 结果写到 `/sdcard/Android/data/ai.mlc.mlcengineexample/files/mlc-bench-result.json`。
5. Mac 侧脚本拉回 JSON，并转换成与 `14-013.md` 一致的 CSV 行。

## 准备源码、打包和安装

当前已经跑通的路线是复用 MLC 官方 Android APK 中的 native runtime，再给 `MLCEngineExample` 加 `BenchmarkActivity`：

```bash
AI/chapter-14/android/mlc/prepare_mlc_android_from_apk.sh
AI/chapter-14/android/mlc/push_mlc_model.sh
```

这条路线会：

- 下载 `binary-mlc-llm-libs` 发布的 `mlc-chat.apk`。
- 提取 `libtvm4j_runtime_packed.so`。
- 反编译 APK 中与该 native runtime 匹配的旧版 `org.apache.tvm` Java binding。
- 构建并安装带 `BenchmarkActivity` 的 debug APK。
- 下载并推送 `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` 到 App internal files 目录。

压测命令：

```bash
MODEL_ID="Qwen2.5-1.5B-Instruct-q4f16_1-MLC" \
MODEL_LIB="qwen2_q4f16_1_2e221f430380225c03990ad24c3d030e" \
MODEL_PATH="/data/data/ai.mlc.mlcengineexample/files/Qwen2.5-1.5B-Instruct-q4f16_1-MLC" \
AI/chapter-14/android/mlc/bench_mlc_android.sh
```

如果要走 MLC 官方 `mlc_llm package` 路线，则使用：

```bash
AI/chapter-14/android/mlc/prepare_mlc_android.sh
```

默认使用：

```text
HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC
```

如果要换模型：

```bash
MLC_MODEL="HF://mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC" \
MLC_MODEL_ID="Qwen2.5-1.5B-Instruct-q4f16_1-MLC" \
MLC_ESTIMATED_VRAM_BYTES=1800000000 \
AI/chapter-14/android/mlc/prepare_mlc_android.sh
```

脚本会拉取 `mlc-ai/mlc-llm` 到 `AI/chapter-14/android/src/mlc-llm`，应用 `patches/0001-add-mlc-benchmark-activity.patch`，执行 `mlc_llm package`，构建 debug APK，并通过 adb 安装。

如果本机已有可用的 MLC Python 环境，但 `mlc_llm` 不在 PATH 中，可以显式指定：

```bash
MLC_LLM_BIN="/path/to/mlc_llm" \
AI/chapter-14/android/mlc/prepare_mlc_android.sh
```

如果只想准备源码和 patch，不打包：

```bash
RUN_MLC_PACKAGE=0 BUILD_APK=0 INSTALL_APK=0 \
AI/chapter-14/android/mlc/prepare_mlc_android.sh
```

## 压测

压测前需要知道 `mlc_llm package` 生成的 `model_lib` 名称，并传给脚本：

```bash
MODEL_ID="Qwen2.5-0.5B-Instruct-q4f16_1-MLC" \
MODEL_LIB="<mlc_llm package 生成的 model_lib>" \
AI/chapter-14/android/mlc/bench_mlc_android.sh
```

输出 CSV 默认在：

```text
AI/chapter-14/android/results/pixel9-mlc-bench-YYYYMMDD-HHMMSS.csv
```

CSV 列与 `AI/chapter-14/14-013.md` 的压测结果表保持一致。

## 注意

- 这个路线不是 HTTP API 压测，而是 App 内部直接调用 `MLCEngine`，因此结果更接近 runtime 本身。
- `MODEL_LIB` 必须和 `mlc_llm package` 生成的系统库名字一致。
- `GPU active(%)` 仍然是通过 `dumpsys gpu` 按 App UID 粗略估算，不是严格 PID 级 GPU utilization。

## 当前验证状态

- `prepare_mlc_android_from_apk.sh` 已成功构建并安装带 `BenchmarkActivity` 的 debug APK。
- `push_mlc_model.sh` 已成功推送 `Qwen2.5-1.5B-Instruct-q4f16_1-MLC`。
- Pixel 9 已通过 adb 授权连接。
- 已产出 MLC 压测结果：`AI/chapter-14/android/results/pixel9-mlc-bench-20260714-041433.csv`。
- 当前机器尝试安装 `mlc-llm-nightly-cpu` / `mlc-ai-nightly-cpu` 后，`import mlc_llm` 在 macOS wheel 上遇到 `libtvm.dylib` / TVM FFI 重复注册问题；因此本轮没有采用 `mlc_llm package` 路线，而是复用官方 APK runtime。
