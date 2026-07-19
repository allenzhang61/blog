# llm-train-cpp

这个目录是 `AI/codes/llm-train-python/` 的 C++ 学习版，目标是把 PyTorch 隐藏的几层机制拆开：

- 自研 `Tensor`
- 自研 `Backend`
- 自研动态图 `Autograd`
- CPU reference backend
- CUDA / Metal backend 占位
- GPT 模型模块
- GPT-2 BPE tokenizer 入口
- 纯 C++ 断言测试

第一阶段只实现 CPU。选择 `cuda` 或 `metal` 会得到明确的“尚未实现”错误。

## 目录结构

```text
include/llm/      # public headers；llm.hpp 只是聚合入口
src/core/         # 基础类型、Device、检查函数
src/tensor/       # Tensor 与动态图 autograd
src/ops/          # 统一算子入口，向下调用 backend / kernels
src/backend/      # BackendRegistry、CPUBackend、CUDABackend、MetalBackend
src/kernels/cpu/  # CPU 算子实现：elementwise、matmul、softmax、layernorm、gelu、embedding
src/model/        # GPT 模型模块
src/data/         # GPT-2 BPE tokenizer 与 DataLoader
src/train/        # AdamW 与训练 / 生成流程
```

## 构建

```bash
cmake -S AI/codes/llm-train-cpp -B /tmp/llm-train-cpp-build
cmake --build /tmp/llm-train-cpp-build
```

## 测试

```bash
/tmp/llm-train-cpp-build/llm_cpp_tests
```

测试使用标准库 `assert` 和项目内轻量检查函数，不依赖 Catch2 / GoogleTest。

## 训练示例

```bash
/tmp/llm-train-cpp-build/train_gpt
```

默认运行 1000 次 optimizer step。

## 文本生成示例

```bash
/tmp/llm-train-cpp-build/generate_text
```

## GPT-2 BPE 资源

`tools/export_gpt2_bpe_samples.py` 会用 Python `tiktoken` 生成 C++ tokenizer 对照样例：

```bash
python AI/codes/llm-train-cpp/tools/export_gpt2_bpe_samples.py
```

当前 C++ tokenizer 优先读取这些 GPT-2 BPE 样例，确保关键样例与 `tiktoken.get_encoding("gpt2")` 对齐；未命中的文本使用 byte fallback，后续可以在同一接口下扩展完整 merge 规则。

## TODO

### 与 Python 版本的训练差异

用小模型跑 1000 次同一批数据训练时，两边 loss 都会下降，但 C++ 版本收敛明显慢于 Python / PyTorch 版本。

对比配置：

```text
vocab_size = 256
context_length = 4
emb_dim = 8
n_heads = 2
n_layers = 1
lr = 1e-2
input = [1, 2, 3, 4]
target = [2, 3, 4, 5]
```

结果：

| step | C++ loss | Python loss |
|---:|---:|---:|
| 0 | 5.57582 | 5.910667 |
| 1 | 5.55579 | 5.662953 |
| 10 | 5.37544 | 3.848826 |
| 100 | 3.45190 | 0.014345 |
| 500 | 0.180981 | 0.001354 |
| 1000 | 0.0660066 | 0.000411 |

结论：C++ 版本目前可以说明训练链路跑通，但还不能认为与 Python / PyTorch 数值等价。

主要原因：

- `ops::batch_matmul` 目前没有实现 backward，attention 中 `q @ k^T` 和 `attention_weight @ v` 的梯度链路不完整。
- `Linear::forward()` 中 bias 现在是直接写入 `y.data()`，没有进入 autograd 计算图，因此 bias 梯度不正确。
- `ops::layernorm` 的 backward 还是简化版，没有完整实现 LayerNorm 对输入、scale、shift 的精确梯度。
- 参数初始化没有严格复刻 PyTorch 的初始化细节。

优先修复顺序：

1. 给 `ops::batch_matmul` 补完整 backward。
2. 把 `Linear` 的 bias 加法改成可追踪的张量操作。
3. 补完整 `LayerNorm` backward。
4. 对齐 PyTorch 初始化方式，再重新做 1000 step 对比。
