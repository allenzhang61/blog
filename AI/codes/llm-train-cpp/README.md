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

注意：这个训练示例是很小的学习模型。Metal 路径目前主要覆盖 forward kernel，backward 和优化器仍使用 host mirror 计算；CUDA 路径已经为训练 smoke path 覆盖 device-resident data/grad、核心 backward kernel、causal mask 和 AdamW 参数更新。也就是说，模型参数相关的训练热路径（参数 data、参数 grad、backward 梯度累积、AdamW 的 m/v 和参数更新）已经基本在 CUDA GPU 上完成；外围的 tokenizer、DataLoader、batch 构造、模型初始化、日志输出，以及 `data()` / `grad()` / `item()` 调试读取仍会经过 CPU/host mirror。

如果想确认 Metal GPU 路径确实在执行，可以用更大的 matmul benchmark 制造持续 GPU 负载：

```bash
/tmp/llm-train-cpp-metal-build/metal_benchmark 1024 100
```

参数含义：

```text
metal_benchmark <matrix_size> <iterations>
```

例如 `1024 100` 表示连续运行 100 次 `1024 x 1024` 的 Metal 矩阵乘法。

## 运行效果对比

当前小模型训练示例的对比结果：

| 实现 | 设备 / 路径 | 1000 steps loss | 当前结论 |
|---|---|---:|---|
| C++ | CPU | 0.23917 | reference backend |
| C++ | Metal | 0.239176 | 与 C++ CPU 基本一致 |
| Python / PyTorch | 默认 PyTorch 路径 | 约 0.099717 | 与当前 C++ 实现不完全一致 |

结论：

- Metal 和 C++ CPU：当前小模型训练结果基本一致，可以说明 Metal backend 的 forward kernel 和 C++ host mirror 训练链路能对齐。
- Metal 和 Python / PyTorch：不完全一致。原因主要是 C++ 自研 `Tensor`、`Autograd`、`AdamW`、初始化和部分算子实现并没有逐项复刻 PyTorch。
- Metal 不是完整 GPU 训练：forward 中的主要算子尽量走 Metal；backward、梯度累积和 AdamW 仍在 host mirror 上执行。
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
- Metal 训练链路可以跑通；forward 尽量走 Metal，backward 和 AdamW 暂时使用 host mirror 计算。
- CUDA backend 覆盖训练 smoke path 的 device-resident forward/backward/AdamW；未覆盖的调试读取仍通过 host mirror 懒同步。
- 无 CUDA Toolkit 或无 CUDA device 的机器会走清晰的 unavailable/skip 路径。
- GPU kernel 目标是学习和验证 backend 分发，不做性能优化。
