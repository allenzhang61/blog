# Qwen3.5-4B 本地推理示例

这个目录演示如何在本地加载 `Qwen/Qwen3.5-4B` 并执行文本生成。当前入口不再兼容其他模型，默认面向 CUDA 设备运行。

默认输入语句：

```text
介绍一下 TCP 三次握手
```

默认模型：

```text
Qwen/Qwen3.5-4B
```

## 安装依赖

```bash
cd AI/codes/llm-inference-python
python -m pip install -r requirements.txt
```

依赖说明：

- `torch`：执行 CUDA 张量计算和模型推理
- `transformers`：加载 Qwen3.5 模型、processor 和 chat template
- `accelerate`：支持低内存加载和 device map
- `huggingface_hub`：下载和缓存 Hugging Face 模型文件
- `safetensors`：读取 safetensors 权重
- `pillow`、`torchvision`：满足 Qwen3.5 multimodal processor 的依赖
- `causal-conv1d`、`flash-linear-attention`：启用 Qwen3.5 linear attention fast path

> 注意：`Qwen/Qwen3.5-4B` 需要较新的 Transformers 版本。若 PyPI 版本暂不支持 `AutoModelForMultimodalLM`，可安装 Transformers main 分支。
> 在 Linux/WSL2 上安装 `causal-conv1d` 时需要匹配 PyTorch CUDA 版本的 `nvcc`。例如 PyTorch `cu128` 对应 CUDA Toolkit 12.8。

## 默认运行

首次运行会从 Hugging Face 下载配置、processor、tokenizer 和权重文件。4B FP16 权重需要数 GB 磁盘空间，首次下载会比较慢。

```bash
python main.py
```

等价于：

```bash
python main.py --prompt '介绍一下 TCP 三次握手' --device cuda --dtype float16
```

## 常用参数

覆盖 prompt：

```bash
python main.py --prompt '解释一下 epoll 的工作原理'
```

指定缓存目录：

```bash
python main.py --cache-dir /data/huggingface-cache
```

使用贪心解码：

```bash
python main.py --greedy
```

限制生成 token 数：

```bash
python main.py --max-new-tokens 64
```

## RTX 3080 运行建议

RTX 3080 通常是 12GB 显存。Windows 设备 `lxa@192.168.1.110` 上建议在 WSL2 `Ubuntu-D` 中运行，并把项目、虚拟环境和模型缓存放在 WSL 的 Linux 文件系统内：

```bash
cd /home/zyl/llm-inference-python
source .venv/bin/activate
python main.py \
  --device cuda \
  --dtype float16 \
  --max-new-tokens 64 \
  --greedy \
  --cache-dir /home/zyl/hf-cache
```

如果显存紧张，可以降低 `--max-new-tokens`，或改用量化/GGUF 版本另行实现加载路径。

## 代码结构

```text
AI/codes/llm-inference-python/
├── main.py                 # Qwen3.5-4B 命令行入口
├── model/                  # 早期自实现 LLM 结构，当前入口不再使用
├── tool/                   # 早期工具模块，当前入口不再使用
└── generate/               # 早期生成模块，当前入口不再使用
```
