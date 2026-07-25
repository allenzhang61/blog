# llm-train-cpp

这个目录是 `AI/codes/llm-train-python/` 的 C++ 学习版，目标是把 PyTorch 隐藏的几层机制拆开：

- 自研 `Tensor`
- 自研 `Backend`
- 自研动态图 `Autograd`
- CPU reference backend
- CUDA / Metal backend 可选接入
- GPT 模型模块
- GPT-2 BPE tokenizer 入口
- 纯 C++ 断言测试

默认构建只启用 CPU。Metal 在 Apple 平台可通过 CMake 选项启用，CUDA 在安装 CUDA Toolkit 的环境中可通过 CMake 选项启用；不可用时会给出明确错误。

## 目录结构

```text
include/llm/      # public headers；llm.hpp 只是聚合入口
include/llm/backend/ # Backend、BackendRegistry、CPUBackend 等按类拆分的头文件
include/llm/data/    # GPT2BPETokenizer、DataLoader
include/llm/model/   # Module、Linear、Embedding、GPTModel 等模型类
include/llm/train/   # AdamW、Trainer
src/core/         # 基础类型、Device、检查函数
src/tensor/       # Tensor 与动态图 autograd
src/ops/          # 统一算子入口，向下调用 backend / kernels
src/backend/      # BackendRegistry、CPUBackend、CUDABackend、MetalBackend 等后端入口
src/kernels/cpu/  # CPU 算子实现：elementwise、matmul、softmax、layernorm、gelu、embedding
src/kernels/metal/# Metal shader 源与 kernel dispatch
src/model/        # GPT 模型模块，按 class 拆分实现
src/data/         # GPT-2 BPE tokenizer 与 DataLoader，按 class 拆分实现
src/train/        # AdamW 与训练 / 生成流程，按 class 拆分实现
```

约定：聚合头文件如 `llm/model.hpp`、`llm/data.hpp`、`llm/train.hpp`、`llm/backend.hpp` 只负责集中 include；具体 class 尽量放在同名头文件和源文件中。同一类模块放到同一子目录下，方便学习时按模块定位。

## 构建

```bash
cmake -S AI/codes/llm-train-cpp -B /tmp/llm-train-cpp-build
cmake --build /tmp/llm-train-cpp-build
```

Metal-enabled build：

```bash
cmake -S AI/codes/llm-train-cpp -B /tmp/llm-train-cpp-metal-build -DLLM_CPP_ENABLE_METAL=ON
cmake --build /tmp/llm-train-cpp-metal-build
```

CUDA-enabled build：

```bash
cmake -S AI/codes/llm-train-cpp -B /tmp/llm-train-cpp-cuda-build -DLLM_CPP_ENABLE_CUDA=ON
cmake --build /tmp/llm-train-cpp-cuda-build
```

Windows + CUDA Toolkit 环境中建议显式选择 Visual Studio generator：

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DLLM_CPP_ENABLE_CUDA=ON
cmake --build build-cuda --config Debug --target llm_cpp_tests llm_cpp_cuda_smoke
ctest --test-dir build-cuda -C Debug --output-on-failure
```

## 测试

```bash
/tmp/llm-train-cpp-build/llm_cpp_tests
```

测试使用标准库 `assert` 和项目内轻量检查函数，不依赖 Catch2 / GoogleTest。

Metal smoke test：

```bash
/tmp/llm-train-cpp-metal-build/llm_cpp_metal_smoke
```

期望输出：

```text
all llm-train-cpp Metal smoke tests passed
```

CUDA smoke test：

```bash
/tmp/llm-train-cpp-cuda-build/llm_cpp_cuda_smoke
```

期望输出：

```text
all llm-train-cpp CUDA smoke tests passed
```

## 训练示例

```bash
/tmp/llm-train-cpp-build/train_gpt
```

默认运行 1000 次 optimizer step。

也可以显式选择 backend：

```bash
LLM_CPP_BACKEND=cpu /tmp/llm-train-cpp-build/train_gpt
/tmp/llm-train-cpp-build/train_gpt cpu
```

当前 `train_gpt` 支持 CPU、Metal 和 CUDA 路径；CUDA 需要 CUDA Toolkit、可用 NVIDIA GPU 和 CUDA-enabled build。

Metal-enabled build 下可以运行：

```bash
/tmp/llm-train-cpp-metal-build/train_gpt metal
```

当前小模型示例中，CPU 和 Metal 1000 步训练结果基本一致：

```text
train_gpt cpu:0 loss after 1000 steps: 0.23917
train_gpt metal:0 loss after 1000 steps: 0.239176
```

注意：这个训练示例是很小的学习模型。Metal 和 CUDA 路径都已经为训练 smoke path 覆盖了 device-resident data/grad、核心 backward kernel 和 AdamW 参数更新。也就是说，模型参数相关的训练热路径（参数 data、参数 grad、backward 梯度累积、AdamW 的 m/v 和参数更新）可以基本在 GPU 上完成；外围的 tokenizer、DataLoader、batch 构造、模型初始化、日志输出，以及 `data()` / `grad()` / `item()` 调试读取仍会经过 CPU/host mirror。

如果想确认 Metal GPU 路径确实在执行，可以用更大的 matmul benchmark 制造持续 GPU 负载：

```bash
/tmp/llm-train-cpp-metal-build/metal_benchmark 1024 100
```

参数含义：

```text
metal_benchmark <matrix_size> <iterations>
```

例如 `1024 100` 表示连续运行 100 次 `1024 x 1024` 的 Metal 矩阵乘法。

更适合做 CPU / Metal / CUDA 横向对比的是通用 benchmark：

```bash
cmake -S AI/codes/llm-train-cpp -B /tmp/llm-train-cpp-cpu-release -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llm-train-cpp-cpu-release --target backend_benchmark
/tmp/llm-train-cpp-cpu-release/backend_benchmark cpu 512 50

cmake -S AI/codes/llm-train-cpp -B /tmp/llm-train-cpp-metal-release -DCMAKE_BUILD_TYPE=Release -DLLM_CPP_ENABLE_METAL=ON
cmake --build /tmp/llm-train-cpp-metal-release --target backend_benchmark
/tmp/llm-train-cpp-metal-release/backend_benchmark metal 512 50
```

Windows + CUDA Toolkit：

```powershell
cmake -S . -B build-cuda-release -G "Visual Studio 17 2022" -A x64 -DLLM_CPP_ENABLE_CUDA=ON
cmake --build build-cuda-release --config Release --target backend_benchmark llm_cpp_cuda_smoke
.\build-cuda-release\Release\llm_cpp_cuda_smoke.exe
.\build-cuda-release\Release\backend_benchmark.exe cuda 512 50
```

本次在 2026-07-24 的测试说明：

- 本机环境：Apple M4 Max，64GB 内存，40 核 Apple GPU，AppleClang 21.0.0。
- CUDA 环境：Windows 机器 `lxa@192.168.1.110`，Intel Core i5-13500H，32GB 内存，NVIDIA GeForce RTX 3080 12GB，driver 610.74，CUDA Toolkit 13.3，MSVC 19.44。
- 测试任务：连续 50 次 `512 x 512` 矩阵乘法；benchmark 会先 warm-up 一次，计时后读取输出以等待 GPU 任务完成。
- CPU backend 是学习用 reference 实现，矩阵乘法没有 BLAS / Accelerate / AMX 优化，所以性能数字主要用于观察当前后端实现差距，而不是代表硬件极限。
- CPU / Metal 跑在本机 Apple M4 Max 上，CUDA 跑在 Windows + RTX 3080 上；相对 CPU 倍数用于感知当前实现差距，严格硬件横评需要在同一台机器上补 CPU baseline。

| Backend | 设备 | 构建类型 | 参数 | 总耗时 | 单次耗时 | 吞吐 | checksum | 相对 CPU |
|---|---|---|---|---:|---:|---:|---:|---:|
| CPU | Apple M4 Max CPU | Release | `512 50` | 10.680184s | 213.603685ms | 1.256699 GFLOPS | -0.084399 | 1.0x |
| Metal | Apple M4 Max 40-core GPU | Release + `LLM_CPP_ENABLE_METAL=ON` | `512 50` | 0.049720s | 0.994403ms | 269.946259 GFLOPS | -0.084399 | 214.8x |
| CUDA | NVIDIA GeForce RTX 3080 12GB | Release + `LLM_CPP_ENABLE_CUDA=ON` | `512 50` | 0.035753s | 0.715056ms | 375.404802 GFLOPS | -0.084399 | 298.7x |

为了让 GPU 运行时间达到 30s 左右，又补了一组更大的长时间 benchmark：

| Backend | 设备 | 构建类型 | 参数 | 总耗时 | 单次耗时 | 吞吐 | checksum |
|---|---|---|---|---:|---:|---:|---:|
| Metal | Apple M4 Max 40-core GPU | Release + `LLM_CPP_ENABLE_METAL=ON` | `2048 1500` | 30.237262s | 20.158175ms | 852.253210 GFLOPS | 6.097171 |
| CUDA | NVIDIA GeForce RTX 3080 12GB | Release + `LLM_CPP_ENABLE_CUDA=ON` | `2048 1000` | 30.942193s | 30.942193ms | 555.224676 GFLOPS | 6.097171 |

当前结论：

- Metal、CUDA 与 CPU 的 checksum 完全一致，说明这组 benchmark 的 matmul 结果对齐。
- 在当前朴素 CPU reference backend 下，Metal 对 `512 x 512` matmul 约快 `214.8x`。
- CUDA 在 RTX 3080 上对这组 matmul 约为 `375.4 GFLOPS`，比本次 Metal 结果快约 `1.39x`。
- 在 `2048 x 2048` 长时间 benchmark 中，Metal 本次约为 `852.3 GFLOPS`，CUDA 本次约为 `555.2 GFLOPS`；这更能压过一次性初始化和小矩阵调度开销。
- 这组结果不能直接推广到完整训练速度：Metal / CUDA 训练链路都已有 device-resident 热路径，但调试读取、外围逻辑、小 kernel 数量和 backward atomic 累加仍会显著影响端到端耗时。

### 训练端到端对比

上面的 matmul benchmark 只看单个核心算子。为了观察完整训练链路，又用默认 `train_gpt` 跑了 1000 steps：

```bash
/usr/bin/time -p /tmp/llm-train-cpp-cpu-release/train_gpt cpu
/usr/bin/time -p /tmp/llm-train-cpp-metal-release/train_gpt metal
```

Windows + CUDA：

```powershell
.\build-cuda-release\Release\train_gpt.exe cuda
```

| Backend | 设备 | steps | loss | 端到端耗时 | 相对 CPU |
|---|---|---:|---:|---:|---:|
| CPU | Apple M4 Max CPU | 1000 | 0.23917 | 24s | 1.0x |
| Metal | Apple M4 Max 40-core GPU | 1000 | 0.239117 | 69s | 0.35x |
| CUDA | NVIDIA GeForce RTX 3080 12GB | 1000 | 0.239129 | 46.88s | 0.51x |

训练端到端结论：

- 三个 backend 的 loss 基本一致，说明这条小模型训练路径结果对齐。
- Metal 已经补了 device-resident forward / backward / AdamW，并把多个 compute encoder 批进 command buffer，只有 host 读取时同步；attention 的 causal mask 也已改为 Metal kernel，不再走 CPU fallback。
- 当前 Metal 训练仍不如 CPU：默认模型太小，backward 里大量使用 atomic 累加 kernel，缺少 kernel fusion，kernel 数量和 command encoding 开销仍然压过了 GPU 并行收益。
- 这个默认训练例子非常小，CPU reference path 反而更快；GPU 版本被大量小 kernel、同步、host/device mirror、数据搬运和训练外围逻辑拖慢。
- 因此当前实现里，GPU backend 更适合用来验证 kernel dispatch 和 device-resident 张量链路；要体现训练吞吐优势，需要更大的 batch / context / hidden size，并继续合并小 kernel、减少 atomic 累加和临时 buffer 分配。

更合理的训练吞吐 benchmark 可以用 `train_benchmark` 指定 batch / context / hidden size：

```bash
cmake --build /tmp/llm-train-cpp-cpu-release --target train_benchmark
cmake --build /tmp/llm-train-cpp-metal-release --target train_benchmark

/tmp/llm-train-cpp-cpu-release/train_benchmark cpu 100 8 16 64 4 1 1024
/tmp/llm-train-cpp-metal-release/train_benchmark metal 100 8 16 64 4 1 1024
```

Windows + CUDA：

```powershell
.\build-cuda-release\Release\train_benchmark.exe cuda 100 8 16 64 4 1 1024
```

参数含义：

```text
train_benchmark <backend> <steps> <batch> <context> <emb> <heads> <layers> <vocab>
```

在 `steps=100 batch=8 context=16 emb=64 heads=4 layers=1 vocab=1024` 下：

| Backend | loss | 耗时 | tokens/s | 相对 CPU |
|---|---:|---:|---:|---:|
| CPU | 4.28624 | 19.0437s | 672.138 | 1.0x |
| Metal | 4.28623 | 1.59187s | 8040.83 | 12.0x |
| CUDA | 4.28623 | 1.56s | 8205.14 | 12.2x |

这组更接近“训练吞吐”场景：每步有 `8 x 16 = 128` 个 token，输出词表扩大到 1024，GPU kernel 的计算量足以压过 command encoding 和小 kernel 调度开销。Metal / CUDA 的 loss 与 CPU 对齐；Metal 吞吐约为 CPU 的 `12.0x`，CUDA 约为 CPU 的 `12.2x`。在这组参数下 CUDA 比 Metal 略快，差距约 `2.0%`，两者基本同一档。

## 运行效果对比

当前小模型训练示例的对比结果：

| 实现 | 设备 / 路径 | 1000 steps loss | 当前结论 |
|---|---|---:|---|
| C++ | CPU | 0.23917 | reference backend |
| C++ | Metal | 0.239176 | 与 C++ CPU 基本一致 |
| Python / PyTorch | 默认 PyTorch 路径 | 约 0.099717 | 与当前 C++ 实现不完全一致 |

结论：

- Metal 和 C++ CPU：当前小模型训练结果基本一致，可以说明 Metal backend 的 forward/backward/AdamW 训练链路能对齐。
- Metal 和 Python / PyTorch：不完全一致。原因主要是 C++ 自研 `Tensor`、`Autograd`、`AdamW`、初始化和部分算子实现并没有逐项复刻 PyTorch。
- Metal 训练路径：forward、核心 backward、梯度累积和 AdamW 已经有 device-resident 实现，但当前没有做性能优化。
- CUDA 训练路径：CUDA Tensor 的 data/grad 在已覆盖算子之间以 device buffer 为权威副本串联，`Tensor::data()`、`Tensor::grad()` 和 `Tensor::item()` 会在 host 读取前同步。
- 因为示例模型很小，`train_gpt metal` 的 GPU usage 不一定明显；观察 GPU 负载建议使用 `metal_benchmark`。

## 文本生成示例

```bash
/tmp/llm-train-cpp-build/generate_text
```

## GPT-2 BPE 资源

`tools/export_gpt2_bpe_ranks.py` 会用 Python `tiktoken` 生成 C++ tokenizer 所需的 GPT-2 BPE rank 表：

```bash
python AI/codes/llm-train-cpp/tools/export_gpt2_bpe_ranks.py
```

C++ tokenizer 读取该 rank 表执行 BPE 合并，与 `tiktoken.get_encoding("gpt2")` 对齐；未命中的文本使用 byte fallback。

## TODO

### 当前限制

- CPU 仍是 reference backend。
- Metal backend 目前覆盖训练 forward 路径中的主要 kernel：elementwise、2D matmul、batch matmul、softmax、layernorm、embedding、GELU、cross entropy。
- Metal 训练链路可以跑通；forward、核心 backward、causal mask 和 AdamW 可以走 Metal device buffer，command buffer 已做批处理，但小模型训练性能仍不理想。
- CUDA backend 覆盖训练 smoke path 的 device-resident forward/backward/AdamW；未覆盖的调试读取仍通过 host mirror 懒同步。
- 无 CUDA Toolkit 或无 CUDA device 的机器会走清晰的 unavailable/skip 路径。
- GPU kernel 目标是学习和验证 backend 分发，不做性能优化。
