# Vast.ai 开发环境配置

本文记录在 Vast.ai GPU 机器上配置 `local-llm` 的开发 / CLion 远端调试环境。

目标状态：

- 远端能用 CMake 编译 `local_llm`
- CLion 能通过 Remote Host 同步源码、构建、运行、调试
- 远端有可运行的 Qwen safetensors 模型目录

## 一、租机器

优先选择 CUDA 开发镜像，而不是只有 runtime 的镜像。至少需要：

- NVIDIA 驱动可用：`nvidia-smi`
- CUDA toolkit 可用：`nvcc --version`
- CMake、g++、gdb、rsync、git
- 开发库：PCRE2、utf8proc

本项目当前默认 CUDA 架构是 `86`，适合 RTX 30 系 / A4000 / A4500 / 3090 这类 Ampere 卡。
如果租到其它架构，构建时用 `-DCMAKE_CUDA_ARCHITECTURES=<sm>` 覆盖。

常见架构：

| GPU | CMAKE_CUDA_ARCHITECTURES |
| --- | --- |
| RTX 3090 / A4000 / A4500 | 86 |
| A100 | 80 |
| H100 / H200 | 90 |
| RTX 4090 / L40 / L40S | 89 |

## 二、连接远端

Vast 控制台会给出 SSH 入口，形如：

```bash
ssh -p <port> root@ssh<id>.vast.ai
```

后续命令中的远端都用这个地址替换。

建议先确认 GPU / CUDA：

```bash
nvidia-smi
which nvcc
nvcc --version
ls -la /usr/local/cuda
```

如果 `nvidia-smi` 正常但找不到 `nvcc`，说明镜像可能只有 runtime，没有 toolkit。
换 CUDA devel 镜像最省事；不建议在临时机器上从零装完整 toolkit，耗时且容易和驱动版本不匹配。

## 三、安装系统依赖

```bash
apt-get update
apt-get install -y \
  build-essential cmake gdb git rsync pkg-config \
  libpcre2-dev libutf8proc-dev \
  python3 python3-pip
```

检查依赖：

```bash
cmake --version
g++ --version
gdb --version
pkg-config --libs libpcre2-8 utf8proc
```

CUDA 路径建议统一成 `/usr/local/cuda`：

```bash
ls /usr/local | grep cuda
```

如果镜像里实际路径是 `/usr/local/cuda-12.4` 这类，可以补一个软链：

```bash
ln -sfn /usr/local/cuda-12.4 /usr/local/cuda
```

## 四、准备源码目录

CLion Remote Host 会自动把本地项目同步到远端。也可以先手动建目录：

```bash
mkdir -p /root/codes
```

推荐远端源码目录：

```text
/root/codes/local-llm
```

CLion 有时会把项目同步到 `/tmp/tmp.xxxxx/local-llm`。这也能用，但 Vast 实例重启或 CLion 重新配置后路径可能变化。
如果要长期复用，优先配置到 `/root/codes/local-llm`。

## 五、配置 CLion Remote Host

### 5.1 Toolchain

CLion 设置：

```text
Settings -> Build, Execution, Deployment -> Toolchains
```

新增 `Remote Host`：

- SSH：使用 Vast 给出的 `root@ssh*.vast.ai:<port>`
- CMake：`/usr/bin/cmake`
- C Compiler：`/usr/bin/gcc`
- C++ Compiler：`/usr/bin/g++`
- Debugger：`/usr/bin/gdb`

如果 CUDA 识别失败，确认远端：

```bash
/usr/local/cuda/bin/nvcc --version
```

### 5.2 Deployment

CLion 设置：

```text
Settings -> Build, Execution, Deployment -> Deployment
```

建议 mapping：

```text
Local path:  <本地仓库>/AI/codes/local-llm
Remote path: /root/codes/local-llm
```

如果继续用 CLion 默认临时目录，也要确认 Run/Debug 里 `--model-dir` 指向远端真实模型目录，而不是本地路径。

### 5.3 CMake Profile

CLion 设置：

```text
Settings -> Build, Execution, Deployment -> CMake
```

选择 Remote Host toolchain，并设置 CMake options：

```bash
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_CUDA_ARCHITECTURES=86
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

如果用 Release 性能测试：

```bash
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=86
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

## 六、远端构建

CLion 里点 Build 即可。也可以 SSH 到远端手动构建：

```bash
cd /root/codes/local-llm
cmake -S . -B cmake-build-debug-remote-host \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

cmake --build cmake-build-debug-remote-host --target local_llm -- -j"$(nproc)"
```

构建产物：

```text
cmake-build-debug-remote-host/local_llm
```

## 七、准备模型文件

本项目 Qwen 路径需要 HuggingFace 原始目录结构，至少包含：

```text
config.json
tokenizer.json
model.safetensors-00001-of-00002.safetensors
model.safetensors-00002-of-00002.safetensors
```

推荐远端目录：

```text
/root/models/Qwen3.5-4B-Base
```

### 7.1 从互联网下载

优先让 Vast 机器直接从模型仓库下载。这样通常比从本地电脑上传更快，也避免占用本地上行带宽。

```bash
pip3 install -U huggingface_hub
mkdir -p /root/models
huggingface-cli download Qwen/Qwen3.5-4B-Base \
  --local-dir /root/models/Qwen3.5-4B-Base \
  --local-dir-use-symlinks False
```

如果模型需要登录：

```bash
huggingface-cli login
```

下载后在远端检查：

```bash
ls -lh /root/models/Qwen3.5-4B-Base
test -f /root/models/Qwen3.5-4B-Base/config.json
test -f /root/models/Qwen3.5-4B-Base/tokenizer.json
ls /root/models/Qwen3.5-4B-Base/*.safetensors
```

如果实际使用的不是公开仓库名 `Qwen/Qwen3.5-4B-Base`，把命令里的模型仓库替换成真实 repo id。

### 7.2 从本地同步

只有在远端不能访问模型仓库、模型不是公开仓库，或者下载速度明显慢于本地上传时，再从本地同步。

如果本地已经有模型：

```bash
rsync -av -e 'ssh -p <port>' \
  <本地模型目录>/Qwen3.5-4B-Base/ \
  root@ssh<id>.vast.ai:/root/models/Qwen3.5-4B-Base/
```

同步后在远端检查：

```bash
ls -lh /root/models/Qwen3.5-4B-Base
test -f /root/models/Qwen3.5-4B-Base/config.json
test -f /root/models/Qwen3.5-4B-Base/tokenizer.json
ls /root/models/Qwen3.5-4B-Base/*.safetensors
```

## 八、运行

远端手动运行：

```bash
cd /root/codes/local-llm/cmake-build-debug-remote-host
PROMPT="法国的首都是" ./local_llm \
  --model qwen \
  --model-dir /root/models/Qwen3.5-4B-Base \
  --max-output-tokens 20
```

采样运行：

```bash
PROMPT="法国的首都是" ./local_llm \
  --model qwen \
  --model-dir /root/models/Qwen3.5-4B-Base \
  --max-output-tokens 40 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --seed 7
```

性能明细：

```bash
PROMPT="法国的首都是" ./local_llm \
  --model qwen \
  --model-dir /root/models/Qwen3.5-4B-Base \
  --max-output-tokens 64 \
  --profile \
  --profile-dir .
```

会生成：

```text
profile_qwen.jsonl
profile_qwen_summary.json
profile_qwen_summary.md
```

## 九、CLion Run/Debug 配置

Run/Debug Configuration：

```text
Target: local_llm
Executable: local_llm
Working directory: <远端 build 目录>
```

Program arguments：

```bash
--model qwen --model-dir /root/models/Qwen3.5-4B-Base --max-output-tokens 20
```

Environment variables：

```text
PROMPT=法国的首都是
```

如果想减少显存常驻权重缓存，可设置：

```text
LOCAL_LLM_CUDA_WEIGHT_POOL_GB=8
```

## 十、常见问题

### 10.1 `gmake: Makefile: No such file or directory`

说明 CLion 直接执行了 build，但远端 build 目录还没有 configure 成功。

处理：

1. 在 CLion 里 `Reload CMake Project`
2. 确认 CMake profile 使用 Remote Host toolchain
3. 手动检查远端 build 目录是否有 `Makefile`

```bash
ls -lh /root/codes/local-llm/cmake-build-debug-remote-host/Makefile
```

没有就重新 configure：

```bash
cmake -S /root/codes/local-llm \
  -B /root/codes/local-llm/cmake-build-debug-remote-host \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### 10.2 `Cannot open /config.json`

通常是 `--model-dir` 为空，程序拼出了 `/config.json`。

处理：

- CLion Run/Debug 的 Program arguments 必须写 `--model-dir /root/models/Qwen3.5-4B-Base`
- 不要写本地路径，例如 `<本地模型目录>/...`
- 远端确认文件存在：

```bash
ls -lh /root/models/Qwen3.5-4B-Base/config.json
```

### 10.3 找不到 CUDA / cuBLAS / NVML

报错类似：

```text
未找到 CUDA toolkit（cublas_v2.h / libcudart / libcublas）
未找到 NVML（libnvidia-ml）
```

处理：

```bash
ls /usr/local/cuda/include/cublas_v2.h
ls /usr/local/cuda/lib64/libcudart.so
ldconfig -p | grep nvidia-ml
```

必要时显式传：

```bash
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

如果镜像没有 CUDA toolkit，优先换 devel 镜像。

### 10.4 找不到 PCRE2 / utf8proc

报错类似：

```text
未找到 PCRE2
未找到 utf8proc
```

处理：

```bash
apt-get update
apt-get install -y libpcre2-dev libutf8proc-dev
```

### 10.5 CLion 代码红线但远端能编译

常见原因是 CLion 本地索引没有拿到远端 CUDA include 或 compile commands。

处理：

1. `Reload CMake Project`
2. 删除远端 build 目录后重新 configure
3. 确认 `compile_commands.json` 在远端 build 目录生成
4. 确认 `.cu` 文件属于 `local_llm` target

### 10.6 SSH / rsync 失败

Vast 实例刚启动时 SSH 可能需要等几十秒。先重试：

```bash
ssh -p <port> root@ssh<id>.vast.ai 'hostname && nvidia-smi'
```

如果 CLion 同步慢，可以先用命令行确认 `rsync` 可用：

```bash
rsync -av -e 'ssh -p <port>' ./ root@ssh<id>.vast.ai:/root/codes/local-llm/
```

## 十一、推荐目录约定

```text
/root/codes/local-llm                  # 源码
/root/codes/local-llm/cmake-build-*    # 构建目录
/root/models/Qwen3.5-4B-Base           # Qwen safetensors 模型
/root/models/*.gguf                    # DeepSeek / llama.cpp GGUF 模型
```

不要把模型放到 CLion 的临时同步目录里；重新同步项目时容易误删或重复上传。
