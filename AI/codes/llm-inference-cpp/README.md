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
