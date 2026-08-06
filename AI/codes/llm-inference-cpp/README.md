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

当前版本只保留 CUDA 推理路径，已能在 RTX 3080 上完成 Qwen3.5-4B greedy 生成：

- 解析 `config.json`
- 扫描 Hugging Face 模型目录中的 `.safetensors`
- 使用 `mmap` 映射 safetensors 文件
- 解析 safetensors header 中的 tensor 名称、dtype、shape、data_offsets
- 读取 BF16 / F32 tensor
- 使用内置默认 prompt token ids 跑 prefill
- CUDA 实现 Qwen3.5 的 linear attention / full attention / MLP / RMSNorm / RoPE / greedy logits
- 输出加载耗时、prefill 耗时、decode 耗时和 generated token ids
- 保留与 Python 入口相似的 CLI 参数

还有这些限制：

- 当前只内置了默认 prompt 的 token ids；任意 prompt 需要继续实现 `tokenizer.json` BPE tokenizer，或者用 `--input-ids` 手动传入 token ids
- 当前只实现 greedy

## 构建

```bash
cmake -S . -B build-cuda128 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-cuda128 -j
```

## 代码结构

```text
src/main.cpp/.h               CLI 流程编排
src/core/cli.cpp/.h           参数解析
src/core/common.cpp/.h        常量、日志、计时、基础类型
src/core/config.cpp/.h        config.json 解析，使用 nlohmann/json
src/core/cuda_kernels.cu/.h   CUDA kernel launch 封装
src/safetensors/safetensors.cpp/.h   safetensors mmap 和 tensor metadata
src/core/tokenizer.cpp/.h     默认 prompt ids、vocab 反查、detokenize
src/core/profile.cpp/.h       JSON timing 和 tensor dump
src/kernels/cuda/cuda_ops.cpp/.h CUDA fused attention / MLP / prefill 操作
src/model/QwenModel.cpp/.h    Qwen3.5 CUDA forward、linear/full attention、KV/recurrent cache
src/model/weights.cpp/.h      Qwen 权重命名和校验
third_party/nlohmann/json.hpp vendored JSON single-header
```

## 运行

模型目录应是 Hugging Face 缓存或下载后的目录，里面至少有：

```text
config.json
*.safetensors
```

例如：

```bash
./build-cuda128/llm-inference-cpp \
  --model-dir /home/zyl/hf-cache/models--Qwen--Qwen3.5-4B/snapshots/<snapshot> \
  --prompt "介绍一下 TCP 三次握手" \
  --max-new-tokens 64 \
  --greedy \
  --profile-timing
```

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

## 性能进展与 llama.cpp 对齐

为了向 llama.cpp BF16 口径靠近，已经做了这些优化：

1. CPU 手写快路径：去掉 matvec 内层循环中的 dtype 字符串判断和函数调用，按 BF16/F16/F32 直接扫指针。
2. CUDA/cuBLAS matvec：使用 CUDA 12.8 + cuBLAS，把 BF16/F32 权重缓存到 GPU VRAM，用 `cublasGemmEx` 执行大矩阵向量乘。
3. fused MLP：`gate_proj/up_proj -> SiLU * up -> down_proj` 中间激活留在 GPU。
4. project attention：linear attention 和 full attention 的 projection、attention core、state/cache、out projection 留在 GPU。
5. concat projection：把同层多路 projection 合并为一次大 matvec，减少 cuBLAS launch 次数。
6. GPU argmax：logits matvec + argmax 在 GPU 上完成，只拷回 token id。
7. full layer pipeline：linear/full attention 层的 input RMSNorm、attention、residual、post RMSNorm、MLP、residual 在 GPU 中串起来，每层只拷回最终 hidden。
8. device hidden + batch prefill：decode 阶段 hidden 在层间常驻 GPU；prefill 阶段把 15 个 prompt token 合成 `[T, hidden]` batch path，projection / MLP 从 GEMV 改为 batched GEMM，linear/full attention 增加 batch prefill CUDA kernel。

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
  -DCMAKE_BUILD_TYPE=Release \
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

当前 64-token 对比，对齐 `../llm-inference-python/doc/5.md` 中 llama.cpp BF16 的测试口径：

| 版本 | 模型格式 | generated ids 是否对齐 | prefill | decode | decode/token | decode speed | infer wall |
|------|----------|------------------------|---------|--------|--------------|--------------|------------|
| llama.cpp CUDA backend | GGUF BF16 | 基准 | 0.0183s | 0.8049s | 12.58ms | 79.5 tok/s | llama-bench 未按同口径拆 |
| Python / Transformers CUDA，预热后 | safetensors FP16 | 是 | 0.0510s | 2.5016s 含 sample | 39.61ms | 25.58 tok/s | 2.56s |
| C++ CUDA project attention + fused MLP | safetensors BF16 | 是 | 0.3540s | 1.5981s | 25.37ms | 39.42 tok/s | 1.9681s |
| C++ CUDA concat projection + project attention + fused MLP | safetensors BF16 | 是 | 0.3238s | 1.4411s | 22.87ms | 43.72 tok/s | 1.7787s |
| C++ CUDA full layer pipeline | safetensors BF16 | 是 | 0.2737s | 1.1981s | 19.02ms | 52.58 tok/s | 1.4852s |
| C++ CUDA device hidden + batch prefill | safetensors BF16 | 是 | 0.0243-0.0248s | 0.9311-0.9342s | 14.55-14.60ms | 68.5-68.7 tok/s | 0.9647-0.9677s |

C++ 版已经超过 Python / Transformers CUDA，但还没有超过 llama.cpp BF16。当前最优 C++ decode 是约 0.932s，llama.cpp 是约 0.801s，还差约 1.16 倍；prefill 已经从旧路径约 0.199s 降到约 0.024s，接近 llama.cpp 的 0.016-0.018s。

CUDA concat projection + project attention + fused MLP 已经把这些路径搬到 GPU：

- MLP 子层：`gate_proj/up_proj -> SiLU * up -> down_proj` 中间激活留在 GPU。
- Linear attention：`in_proj_qkv/z/b/a -> conv1d -> recurrent state -> gated RMSNorm -> out_proj` 留在 GPU，conv/recurrent state 常驻 VRAM。
- Full attention：`q/k/v projection -> q/k RMSNorm -> RoPE -> KV cache -> softmax attention -> o_proj` 留在 GPU，KV cache 常驻 VRAM。
- Logits：embedding matvec + argmax 在 GPU 上完成，只拷回 token id。
- 同层 projection 合并：linear attention 的 `qkv/z/b/a`、full attention 的 `q/k/v`、MLP 的 `gate/up` 合并为更大的 cuBLAS matvec。
- 整层 pipeline：attention 输出不再回 CPU 后再进入 MLP，而是在 GPU 内完成 residual、post RMSNorm、MLP 和第二次 residual。
- 层间 device hidden：decode 阶段不再每层把 hidden 拷回 CPU，整段 greedy 生成也把 token id 留在 GPU，最后一次性拷回 generated ids。
- batch prefill：prefill 不再对 15 个 prompt token 顺序执行 15 次完整 forward，而是一次性处理 `[T, hidden]`，并在层内批量更新 linear attention recurrent state / full attention KV cache。

64-token 正式推理已经从 Python / Transformers CUDA 的 2.56s 降到约 0.965s，并且 generated ids 对齐。

为了继续追 llama.cpp，剩余关键点不是 Python/C++ 调度，而是 GEMV backend：

- 当前主要矩阵乘仍是 cuBLAS `m x k` by `k x 1` 的 GEMV/GEMM 形态，batch=1 时对 RTX 3080 的内存带宽利用不如 llama.cpp 的专用 CUDA kernel。这是当前最大差距。
- 每 token 仍有大量小 GEMV，尽管已经合并了一部分 projection。
- prefill 已改成 batch path，但 batch path 里 linear/full attention projection 还没有完全复用 decode 的 concat projection 布局，仍有继续融合空间。
- decode 已让 hidden 在层间常驻 GPU，剩余差距主要来自 batch=1 GEMV backend、kernel launch 数量、权重/KV layout 和专用 fused kernel。
- 历史实验：`LLM_INFERENCE_CUDA_FUSE_RMSNORM_MLP=1` 曾用于融合 `post_attention_layernorm + MLP`，本轮 64-token 实测为 15.29s，慢于默认路径，代码中已移除。
- 历史实验：`LLM_INFERENCE_CUDA_CONVERT_BF16_TO_F16=1` 曾用于把 2D BF16 权重入 VRAM 时转成 FP16；8-token 对齐，但 warmup 从约 8s 增加到约 68s，正式推理几乎无收益，代码中已移除。
- 历史实验：`LLM_INFERENCE_CUDA_CUSTOM_BF16_GEMV=1` 曾用于启用原生 one-block-per-row BF16 GEMV；8-token 对齐，但从约 0.51s 退到约 0.61s，慢于 cuBLAS，代码中已移除。
