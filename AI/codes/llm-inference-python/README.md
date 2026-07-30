# 自实现 LLM forward 推理示例

这个目录演示如何用 Python 写一个简洁的 LLM forward 推理流程。它不是 Ollama API 封装，也不会调用远程 HTTP 推理服务；Hugging Face 只用于下载和缓存模型文件，tokenizer 只用于文本和 token id 的转换，模型 forward 路径由本目录代码实现。

默认输入语句：

```text
法国的首都是
```

默认模型标识：

```text
deepseek-r1:8b
```

默认 Hugging Face repo：

```text
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## 安装依赖

```bash
cd AI/codes/llm-inference-python
python -m pip install -r requirements.txt
```

依赖说明：

- `torch`：执行张量计算和自实现 forward
- `huggingface_hub`：下载和缓存 Hugging Face 模型文件
- `safetensors`：读取 safetensors 权重
- `transformers`：只用于加载 tokenizer，不使用 `AutoModelForCausalLM.generate()`

## 默认运行

首次运行会从 Hugging Face 下载配置、tokenizer 和权重文件。8B 权重体积较大，需要较长下载时间和足够磁盘空间。

```bash
python main.py
```

等价于：

```bash
python main.py --prompt '法国的首都是' --model deepseek-r1:8b
```

## 覆盖 prompt

```bash
python main.py '北京的首都是'
```

或者：

```bash
python main.py --prompt '北京的首都是'
```

## 覆盖模型和 Hugging Face repo

`--model` 是本地 CLI 展示用的模型标识。`--repo-id` 用于指定实际下载的 Hugging Face repo。

```bash
python main.py \
  --model deepseek-r1:8b \
  --repo-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

也可以指定 revision：

```bash
python main.py --revision main
```

## 指定缓存目录

```bash
python main.py --cache-dir /data/huggingface-cache
```

重复运行时，`huggingface_hub` 会复用本地缓存，已有文件不会重复下载。

## 使用本地模型目录

如果模型已经下载好，可以绕过 Hugging Face 下载：

```bash
python main.py --model-dir /path/to/DeepSeek-R1-Distill-Llama-8B
```

目录中至少需要：

```text
config.json
tokenizer.json 或 tokenizer.model
*.safetensors 或 pytorch_model*.bin
```

## 生成参数

```bash
python main.py \
  --prompt '法国的首都是' \
  --max-new-tokens 32 \
  --temperature 0.7
```

使用贪心解码：

```bash
python main.py --greedy
```

指定设备：

```bash
python main.py --device cuda
python main.py --device cpu
python main.py --device mps
```

## 常见问题

### 网络或鉴权失败

如果 Hugging Face 下载失败，检查：

- 网络是否能访问 Hugging Face
- `--repo-id` 是否正确
- `--revision` 是否存在
- 私有或 gated 模型是否已经登录并配置 token

### 显存不足

`deepseek-r1:8b` 是 8B 级别模型，完整权重和激活会占用较多显存。本示例没有实现量化、KV Cache、分页显存管理等生产优化。显存不足时可以：

- 改用 CPU 运行，但速度会慢
- 降低 `--max-new-tokens`
- 后续扩展量化加载
- 使用更小且结构兼容的模型 repo

### 权重不兼容

本示例实现的是简洁 Llama-like decoder-only Transformer。若权重 key 或形状不匹配，脚本会输出缺失 key 或 shape mismatch 信息。不同模型家族可能需要调整权重映射和模型结构。

## 代码结构

目录结构参考 `AI/codes/llm-train-python/` 拆分：

```text
AI/codes/llm-inference-python/
├── main.py                 # 命令行入口
├── model/                  # 自实现 LLM 模型结构
│   ├── LLMConfig.py
│   ├── RMSNorm.py
│   ├── RotaryEmbedding.py
│   ├── SelfAttention.py
│   ├── MLP.py
│   ├── DecoderLayer.py
│   └── SimpleLLM.py
├── tool/                   # 依赖、下载、tokenizer、权重、运行设备
│   ├── model_files.py
│   ├── model_loader.py
│   ├── runtime.py
│   ├── tokenizer.py
│   └── weights.py
└── generate/
    └── generate.py         # 逐 token 生成和采样
```

主要流程：

```text
prompt -> tokenizer -> token ids
      -> embedding
      -> Transformer blocks
      -> final RMSNorm
      -> LM head logits
      -> greedy / temperature sampling
      -> decode text
```
