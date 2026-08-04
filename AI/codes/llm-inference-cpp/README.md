# llm-inference-cpp

参考 `../llm-inference-python/main.py` 开始做的 C++ 学习版实现。

这个目录不引用 llama.cpp，也不通过 llama.cpp API 间接完成推理。目标是从 0 实现 Qwen3.5-4B 推理所需的基础模块，逐步对齐 Python 版本里的：

- 模型加载耗时
- prompt/token 输入
- prefill 耗时
- decode 耗时
- KV cache
- warmup 后统计

## 当前进度

当前版本已经能在 CPU 上完整跑通一次 Qwen3.5-4B 文本 forward：

- 解析 `config.json`
- 扫描 Hugging Face 模型目录中的 `.safetensors`
- 使用 `mmap` 映射 safetensors 文件
- 解析 safetensors header 中的 tensor 名称、dtype、shape、data_offsets
- 读取 BF16 / F32 tensor
- 使用内置默认 prompt token ids 跑 prefill
- 实现 Qwen3.5 的 linear attention / full attention / MLP / RMSNorm / RoPE / greedy logits
- 输出加载耗时、prefill 耗时、logits 耗时和 generated token ids
- 保留与 Python 入口相似的 CLI 参数

还有这些限制：

- 当前只内置了默认 prompt 的 token ids；任意 prompt 需要继续实现 `tokenizer.json` BPE tokenizer，或者用 `--input-ids` 手动传入 token ids
- 当前是 CPU 教学实现，速度很慢
- 当前只实现 greedy
- CUDA kernel 还没有实现

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

如果环境里没有 `cmake`，也可以直接用：

```bash
mkdir -p build
g++ -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Wpedantic \
  -Iinclude \
  src/core/*.cpp \
  src/model/*.cpp \
  src/main.cpp \
  -o build/llm-inference-cpp
```

## 代码结构

```text
include/llm_inference.h       公共结构、函数声明
src/main.cpp                  CLI 流程编排
src/core/cli.cpp              参数解析
src/core/common.cpp           常量、日志、计时
src/core/config.cpp           config.json 解析
src/core/safetensors.cpp      safetensors mmap 和 tensor metadata
src/core/tensor_ops.cpp       BF16/F16/F32 读取、matvec、norm、激活函数
src/core/tokenizer.cpp        默认 prompt ids、vocab 反查、detokenize
src/core/profile.cpp          JSON timing 和 tensor dump
src/model/NativeQwen.cpp      Qwen3.5 forward、linear/full attention、KV/recurrent cache
```

## 运行

模型目录应是 Hugging Face 缓存或下载后的目录，里面至少有：

```text
config.json
*.safetensors
```

例如：

```bash
./build/llm-inference-cpp \
  --model-dir /home/zyl/hf-cache/models--Qwen--Qwen3.5-4B/snapshots/<snapshot> \
  --prompt "介绍一下 TCP 三次握手" \
  --max-new-tokens 1 \
  --greedy \
  --profile-timing
```

在 RTX 3080 那台 WSL 机器上，CPU 跑 15 token prefill + 1 token logits 的一次验证耗时约 86.9s，其中 safetensors mmap 加载只有约 0.02s，主要时间都在原生 CPU 矩阵计算。

## 验证结果

在 `lxa@192.168.1.110` 的 `Ubuntu-D` 里，使用 Hugging Face safetensors snapshot：

```text
/home/zyl/hf-cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

原生 C++ CPU 版生成 2 个 token：

```text
generated_ids: [8160, 579]
stdout: Here's
prefill_s: 95.753184
decode_total_s: 6.799077
infer_wall_s: 103.381153
```

Python / Transformers 同样参数的前 2 个 token 也是：

```text
[8160, 579] "Here's"
```

这说明当前 C++ 版的 prefill、greedy logits、一次 decode cache 路径已经和 Python 结果对齐。

## 性能进展

为了向 Python CUDA 版本靠近，已经做了两轮优化：

1. CPU 手写快路径：去掉 matvec 内层循环中的 dtype 字符串判断和函数调用，按 BF16/F16/F32 直接扫指针。
2. CUDA/cuBLAS matvec：使用 CUDA 12.8 + cuBLAS，把 BF16/F32 权重缓存到 GPU VRAM，用 `cublasGemmEx` 执行大矩阵向量乘。

远端 `Ubuntu-D` 中已安装：

```text
cmake
build-essential
libopenblas-dev
cuda-nvcc-12-8
cuda-cudart-dev-12-8
libcublas-dev-12-8
```

构建 CUDA 版：

```bash
cmake -S . -B build-cuda128 \
  -DLLM_INFERENCE_ENABLE_CUDA=ON \
  -DLLM_INFERENCE_ENABLE_CBLAS=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-cuda128 -j
```

运行 CUDA 版：

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
LLM_INFERENCE_CUDA_WEIGHT_CACHE_GB=10 \
OMP_NUM_THREADS=4 \
./build-cuda128/llm-inference-cpp \
  --model-dir /home/zyl/hf-cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --max-new-tokens 64 \
  --greedy \
  --profile-timing \
  --warmup-runs 1
```

当前 64-token 对比：

| 版本 | generated ids 是否对齐 | infer wall | prefill | decode |
|------|------------------------|------------|---------|--------|
| Python / Transformers CUDA，预热后 | 是 | 2.56s | 0.051s | 2.50s |
| C++ 原生 CPU，优化前 | 是 | 136.29s | 87.94s | 47.50s |
| C++ 原生 CPU，dtype 快路径 | 是 | 52.62s | 32.07s | 20.00s |
| C++ 原生 CUDA/cuBLAS matvec，预热后 | 是 | 16.53s | 2.96s | 13.55s |
| C++ 原生 CUDA/cuBLAS matvec + fused MLP，预热后 | 是 | 15.07s | 2.68s | 12.37s |
| C++ 原生 CUDA project attention + fused MLP，预热后 | 是 | 1.97s | 0.354s | 1.598s |

CUDA project attention + fused MLP 已经把这些路径搬到 GPU：

- MLP 子层：`gate_proj/up_proj -> SiLU * up -> down_proj` 中间激活留在 GPU。
- Linear attention：`in_proj_qkv/z/b/a -> conv1d -> recurrent state -> gated RMSNorm -> out_proj` 留在 GPU，conv/recurrent state 常驻 VRAM。
- Full attention：`q/k/v projection -> q/k RMSNorm -> RoPE -> KV cache -> softmax attention -> o_proj` 留在 GPU，KV cache 常驻 VRAM。
- Logits：embedding matvec + argmax 在 GPU 上完成，只拷回 token id。

64-token 正式推理已经从 Python / Transformers CUDA 的 2.56s 降到 1.97s，并且 generated ids 对齐。

剩余可继续优化的点：

- `input_layernorm`、`post_attention_layernorm` 和 residual add 仍在 CPU 上。
- 每层 mixer/MLP 输出仍会回 CPU 做 residual。
- prefill 仍按 token 逐个执行，没有像 Transformers 那样批量处理 15 token。
- 另有 `LLM_INFERENCE_CUDA_FUSE_RMSNORM_MLP=1` 可实验性融合 `post_attention_layernorm + MLP`，本轮 64-token 实测为 15.29s，慢于默认路径，所以默认关闭。
