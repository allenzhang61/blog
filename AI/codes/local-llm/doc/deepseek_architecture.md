# DeepSeek-V2-Lite 模型架构

本文根据 GGUF 元数据以及本项目的
[`DeepseekConfig.cpp`](../src/llm/model/deepseek/DeepseekConfig.cpp)、
[`DeepseekWeights.h`](../src/llm/model/deepseek/DeepseekWeights.h)
描述当前实现中的 DeepSeek 模型结构。当前代码主要面向 `deepseek2` 架构的
DeepSeek-V2-Lite：注意力使用 MLA，FFN 使用前若干层 dense + 后续 MoE 的结构，
模型文件通过 `MF` 抽象访问，实际权重主要来自 GGUF。

## 一、总体结构

```
              ┌──────────────── GGUF / MF ────────────────┐
              │                                            │
        token_embd → 27 × Decoder Block → output_norm → lm_head
                         │
                         ├─ MLA attention
                         └─ Dense FFN / MoE FFN
```

当前前向路径：

```
embedding
  → for each layer:
      MLA + residual
      MLP(DenseFFN 或 MoE) + residual
  → output_norm
  → output.weight / token_embd fallback
  → sampler
```

- **模型格式**：当前 DeepSeek 路径主要读取 GGUF 的 `deepseek2.*` metadata 与 tensor。
- **注意力**：所有层使用 MLA（Multi-head Latent Attention），KV cache 保存 latent 表示。
- **FFN**：前 `leading_dense_block_count` 层为 dense SwiGLU，后续为 MoE。
- **lm_head**：优先使用独立 `output.weight`，缺失时回退到 `token_embd.weight`；由通用 `common::LMHead` 统一实现（与 Qwen 共用），`vocab_size` 由 Model 外部传入。

## 二、全局超参（DeepseekConfig）

当前默认值与 DeepSeek-V2-Lite 常见 GGUF 配置一致，构造时会从 `MF::metadata()` 读取。

| 参数 | 典型值 | 来源 | 说明 |
| --- | --- | --- | --- |
| `hidden_size` | 2048 | `deepseek2.embedding_length` | Transformer 隐藏维度 |
| `num_layers` | 27 | `deepseek2.block_count` | Decoder block 层数 |
| `vocab_size` | 102400 | `deepseek2.vocab_size` | 词表大小 |
| `num_heads` | 16 | `deepseek2.attention.head_count` | MLA 中的 query head 数 |
| `rms_norm_eps` | 1e-6 | `deepseek2.attention.layer_norm_rms_epsilon` | RMSNorm epsilon |
| `eos_token_id` | 模型文件决定 | `tokenizer.ggml.eos_token_id` | 生成终止 token |
| `bos_token_id` | 模型文件决定 | `tokenizer.ggml.bos_token_id` | 起始 token |

> 这里的 `DeepseekConfig` 不是完整 metadata dump，而是当前推理实现需要的结构参数集合。

## 三、MLA Attention

MLA 将 KV 压到较小的 latent 表示中缓存，解码时再通过 `kv_b` 展开出参与注意力计算的
K/V 表示。当前实现中每层都走 MLA。

### 3.1 维度参数

| 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `kv_lora_rank` | 512 | latent KV 维度 |
| `qk_nope_head_dim` | 128 | Q/K 中不参与 RoPE 的部分 |
| `qk_rope_head_dim` | 64 | Q/K 中参与 RoPE 的部分 |
| `qk_head_dim()` | 192 | `qk_nope_head_dim + qk_rope_head_dim` |
| `v_head_dim` | 128 | 每个 value head 的维度 |
| `kv_total` | 576 | `kv_lora_rank + qk_rope_head_dim` |

### 3.2 MLA 子层流程

```
hidden
  → attn_norm
  → attn_q              → q_rope
  → attn_kv_a_mqa       → [latent, k_rope] → latent KV cache
  → attn_kv_b(latent)   → expanded K/V
  → attention(q, K/V, cache)
  → attn_output
  → residual add
```

对应权重（见 `DeepseekLayerWeights`）：

- `attn_norm`：注意力前 RMSNorm；
- `attn_q`：生成 Q，输出维度为 `num_heads * qk_head_dim`；
- `attn_kv_a_mqa`：生成 latent KV 与共享的 RoPE K；
- `attn_kv_a_norm`：对 latent KV 做归一化；
- `attn_kv_b`：把 latent KV 展开为每个 head 的 non-RoPE K 与 V；
- `attn_output`：注意力输出投影回 hidden。

### 3.3 Latent KV Cache

`DeepseekSession` 为每层分配一份 latent KV cache（由类外的 `LatentKVCache` 结构描述）：

```
[max_seq_len, kv_lora_rank + qk_rope_head_dim]
```

也就是典型配置下每个 token 每层缓存 `512 + 64 = 576` 个 `float`。相比缓存完整 K/V，
MLA 通过 latent cache 降低了长上下文推理的 KV 显存占用。

## 四、RoPE / YARN

DeepSeek-V2-Lite 使用 YARN 扩展上下文。当前配置读取：

| 参数 | 典型值 | 来源 |
| --- | --- | --- |
| `rope_dim` | 64 | `deepseek2.rope.dimension_count` |
| `rope_theta` | 10000 | `deepseek2.rope.freq_base` |
| `use_yarn` | true | `deepseek2.rope.scaling.type == "yarn"` |
| `yarn_scaling_factor` | 40 | `deepseek2.rope.scaling.factor` |
| `yarn_original_context` | 4096 | `deepseek2.rope.scaling.original_context_length` |
| `yarn_beta_fast` | 32 | 当前默认值 |
| `yarn_beta_slow` | 1 | 当前默认值 |

`DeepseekSession` 在请求开始时根据这些参数生成 device 端 `inv_freq`，供 MLA RoPE kernel 使用。
当前实现的 attention softmax scale 为：

```
1 / sqrt(qk_head_dim)
```

没有额外把 YARN mscale 乘进 softmax scale。

## 五、FFN：Dense 与 MoE

DeepSeek 的 FFN 子层由 `MLP` 分发：

```
if layer < first_k_dense:
    DenseFFN
else:
    MoE
```

### 5.1 Dense FFN

前 `first_k_dense` 层使用标准 SwiGLU：

```
down( SiLU(gate(x)) * up(x) )
```

| 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `first_k_dense` | 1 | 前多少层使用 dense FFN |
| `dense_ffn` | 10944 | dense FFN 中间维 |

对应权重：

- `ffn_norm`：FFN 前 RMSNorm；
- `ffn_gate` / `ffn_up`：`hidden_size → dense_ffn`；
- `ffn_down`：`dense_ffn → hidden_size`。

### 5.2 MoE FFN

后续层使用 MoE，由三个模块组成：

```
MoERouter → RoutedExperts
          → SharedExperts
          → residual add
```

| 参数 | 典型值 | 说明 |
| --- | --- | --- |
| `expert_count` | 64 | routed expert 总数 |
| `expert_used` | 6 | 每个 token 选择 top-k expert |
| `expert_shared` | 2 | shared expert 数 |
| `expert_ffn` | 1408 | 单个 expert 中间维 |
| `shared_ffn()` | 2816 | `expert_shared * expert_ffn` |
| `routed_scaling` | 1.0 | router 权重缩放 |
| `norm_topk_prob` | false | 当前写死为 false |

#### Router

`MoERouter` 使用 `ffn_gate_inp.weight` 计算 `[tokens, expert_count]` router logits，
然后选出每个 token 的 top-k expert id 与权重。

#### Routed Experts

`RoutedExperts` 按 expert id 从 packed expert tensor 中切出单个 expert 的 `TensorView`，
再通过通用 `CudaWeightPool::cached_weight()` 上传/缓存：

- `ffn_gate_exps`；
- `ffn_up_exps`；
- `ffn_down_exps`。

每个 routed expert 内部仍然是 SwiGLU：

```
expert_down( SiLU(expert_gate(x)) * expert_up(x) )
```

输出乘以 router 权重后累加到 `moe_out`。

#### Shared Experts

`SharedExperts` 不经过 router 选择，对每个 token 都执行 shared SwiGLU：

- `ffn_gate_shexp`；
- `ffn_up_shexp`；
- `ffn_down_shexp`。

shared expert 输出直接累加到 `moe_out`。

## 六、权重清单（DeepseekWeights）

顶层权重：

- `token_embd.weight`：词嵌入；
- `output_norm.weight`：输出前 RMSNorm；
- `output.weight`：lm_head，缺失时回退到 `token_embd.weight`。

每层权重：

- 归一化：`blk.i.attn_norm.weight`、`blk.i.ffn_norm.weight`；
- MLA：`blk.i.attn_q.weight`、`blk.i.attn_kv_a_mqa.weight`、
  `blk.i.attn_kv_a_norm.weight`、`blk.i.attn_kv_b.weight`、`blk.i.attn_output.weight`；
- dense FFN：`blk.i.ffn_gate.weight`、`blk.i.ffn_up.weight`、`blk.i.ffn_down.weight`；
- MoE：`blk.i.ffn_gate_inp.weight`、`blk.i.ffn_gate_exps.weight`、
  `blk.i.ffn_up_exps.weight`、`blk.i.ffn_down_exps.weight`、
  `blk.i.ffn_gate_shexp.weight`、`blk.i.ffn_up_shexp.weight`、
  `blk.i.ffn_down_shexp.weight`。

其中 dense FFN 与 MoE FFN 按层互斥：`layer < first_k_dense` 使用 dense，其余层使用 MoE。

## 七、与本项目实现的对应关系

- 模型文件抽象：[`MF.h`](../src/format/MF.h) 统一提供 metadata、tensor、tokenizer 访问。
- GGUF 读取：[`GgufFile.h`](../src/format/gguf/GgufFile.h) / [`GgufFile.cpp`](../src/format/gguf/GgufFile.cpp)。
- tokenizer：[`GGUFTokenizer.h`](../src/format/gguf/GGUFTokenizer.h) / [`GGUFTokenizer.cpp`](../src/format/gguf/GGUFTokenizer.cpp)。
- 配置解析：[`DeepseekConfig.cpp`](../src/llm/model/deepseek/DeepseekConfig.cpp)。
- 权重索引：[`DeepseekWeights.cpp`](../src/llm/model/deepseek/DeepseekWeights.cpp)。
- 请求状态：[`DeepseekSession.h`](../src/llm/model/deepseek/DeepseekSession.h)，包含 latent KV cache、`inv_freq`、`attn_softmax_scale`；scratch、已生成 token（`outputs`）、`max_seq_len` 上提到公共基类 [`SessionBase.h`](../src/backend/cuda/mem/SessionBase.h)。
- 整体前向：[`DeepseekModel.cpp`](../src/llm/model/deepseek/DeepseekModel.cpp)（prefill / decode 前向路径）。
- 输出头：[`LMHead`](../src/llm/module/common/LMHead.h)（`common::LMHead`，Qwen / DeepSeek 共用）。
- MLA：[`MLA.cpp`](../src/llm/module/deepseek/MLA.cpp)。
- FFN 分发：[`MLP.cpp`](../src/llm/module/deepseek/MLP.cpp)。
- Dense FFN：[`DenseFFN.cpp`](../src/llm/module/deepseek/DenseFFN.cpp)。
- MoE：[`MoE.cpp`](../src/llm/module/deepseek/MoE.cpp)、
  [`MoERouter.cpp`](../src/llm/module/deepseek/MoERouter.cpp)、
  [`RoutedExperts.cpp`](../src/llm/module/deepseek/RoutedExperts.cpp)、
  [`SharedExperts.cpp`](../src/llm/module/deepseek/SharedExperts.cpp)。

## 八、当前实现边界

- 当前 DeepSeek 路径主要按 GGUF `deepseek2` metadata 解析；HF DeepSeek 配置并未完整适配。
- `DeepseekConfig` 只保留推理用到的字段，不保存 GGUF 中所有 metadata。
- `norm_topk_prob` 当前固定为 `false`，没有从模型文件动态读取。
- `yarn_beta_fast` / `yarn_beta_slow` 当前使用默认值，未从 metadata 读取。
- MoE routed expert 在 host 侧取回 top-k expert id/weight 后逐 token、逐 expert 执行，后续仍有融合优化空间。
