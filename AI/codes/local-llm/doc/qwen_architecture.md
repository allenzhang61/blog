# Qwen3.5-4B-Base 模型架构

本文根据 [`config.json`](../../../../../software/llm/Qwen3.5-4B-Base/config.json) 以及本项目的
[`QwenConfig.cpp`](../src/llm/model/qwen/QwenConfig.cpp)、[`QwenWeights.h`](../src/llm/model/qwen/QwenWeights.h)
描述该模型的整体架构。模型顶层 `architectures` 为 `Qwen3_5ForConditionalGeneration`，
`model_type` 为 `qwen3_5`，是一个包含文本塔与视觉塔的多模态模型。本项目当前只走**纯文本推理**路径，
视觉塔（`vision_config` / `model.visual.*`）与 MTP（`mtp.*`）权重虽被解析，但不参与前向。

## 一、总体结构

```
              ┌───────────────── config.json ─────────────────┐
              │                                                │
        text_config (文本塔，实际推理)          vision_config (视觉塔，未使用)
              │                                                │
   embed_tokens → 32 × DecoderLayer → final_norm → lm_head     (24 × ViT block)
```

- **文本塔**：32 层 Decoder，混合两种注意力（linear attention + full attention），FFN 为 SwiGLU MLP。
- **视觉塔**：24 层 ViT，用于图像/视频编码；当前实现不加载进前向。
- **MTP**：1 层多 token 预测头，用于加速解码；当前实现不启用。

## 二、文本塔（text_config）

### 2.1 全局超参

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `hidden_size` | 2560 | Transformer 隐藏维度 |
| `num_hidden_layers` | 32 | Decoder 层数 |
| `intermediate_size` | 9216 | MLP 中间维度（SwiGLU） |
| `vocab_size` | 248320 | 词表大小 |
| `max_position_embeddings` | 262144 | 最大上下文长度 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `hidden_act` | silu | 激活函数（SwiGLU 中的门控用 SiLU） |
| `dtype` | bfloat16 | 权重精度 |
| `tie_word_embeddings` | true | 输入 embedding 与输出 lm_head 权重共享 |
| `eos_token_id` | 248044 | 生成终止 token |

### 2.2 层类型排布（混合注意力）

`layer_types` 定义了 32 层每层的注意力类型，规律是**每 4 层里 3 层 linear + 1 层 full**
（`full_attention_interval = 4`）：

```
层号:  0   1   2   3   4   5   6   7   ...  28  29  30  31
类型: lin lin lin FULL lin lin lin FULL ... lin lin lin FULL
```

即第 3、7、11、15、19、23、27、31 层（下标从 0 起，每第 4 层）为 `full_attention`，
共 8 层 full attention、24 层 linear attention。

### 2.3 Full Attention 层

标准分组查询注意力（GQA）+ QK-Norm + RoPE：

| 参数 | 值 |
| --- | --- |
| `num_attention_heads`（Q heads） | 16 |
| `num_key_value_heads`（KV heads） | 4 |
| `head_dim` | 256 |
| `attn_output_gate` | true（输出带门控） |
| `attention_bias` | false |

对应权重（见 `FullAttnWeights`）：`q_proj` / `k_proj` / `v_proj` / `o_proj`，
以及 `q_norm` / `k_norm`（对 Q、K 做 RMSNorm）。

> 注意：Q 有 16 个 head，KV 只有 4 个 head，每个 KV head 被 4 个 Q head 共享（GQA）。
> `head_dim = 256`，与 `hidden_size / num_attention_heads = 160` 不同，说明 head 维度是独立配置的。

### 2.4 Linear Attention 层（线性/门控注意力，类 Mamba/GDN）

线性注意力层是该模型的主体（24/32 层），带前置深度卷积和门控状态更新：

| 参数 | 值 |
| --- | --- |
| `linear_num_key_heads` | 16 |
| `linear_key_head_dim` | 128 |
| `linear_num_value_heads` | 32 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4（前置 depthwise Conv1d 的 kernel 大小） |
| `mamba_ssm_dtype` | float32（SSM 状态用 fp32 计算） |

对应权重（见 `LinearAttnWeights`）：

- `in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a`：输入投影，分别产生 Q/K/V、门控 z、以及状态更新相关的 b、a 参数；
- `conv1d`：作用在序列维度上的深度卷积（kernel=4）；
- `a_log` / `dt_bias`：SSM 递归衰减/步长相关参数；
- `norm`：层内归一化；
- `out_proj`：输出投影。

> 这是一种线性复杂度的序列建模模块（门控 delta / 状态空间风格），推理时维护 recurrent state 而非 KV cache，
> 与 full attention 层交替，兼顾长上下文效率与表达能力。

### 2.5 MLP（SwiGLU）

每层 FFN 为 SwiGLU 结构（见 `MlpWeights`）：

```
down( SiLU(gate(x)) * up(x) )
```

- `gate` / `up`：`hidden_size(2560) → intermediate_size(9216)`；
- `down`：`intermediate_size(9216) → hidden_size(2560)`。

### 2.6 归一化与残差

每个 Decoder 层包含两处 RMSNorm（见 `LayerWeights`）：

- `input_norm`：注意力子层前的 pre-norm；
- `post_norm`：MLP 子层前的 pre-norm。

塔尾接 `final_norm`（RMSNorm）后再经 lm_head 输出 logits；lm_head 由通用 `common::LMHead` 统一实现（与 DeepSeek 共用），因 `tie_word_embeddings=true` 复用 `embed_tokens` 权重，`vocab_size` 由 Model 外部传入。

### 2.7 位置编码（RoPE）

`rope_parameters`：

| 参数 | 值 | 是否使用 |
| --- | --- | --- |
| `rope_theta` | 1e7 | 使用 |
| `partial_rotary_factor` | 0.25 | 使用（仅对 head_dim 的 25% 施加旋转） |
| `rope_type` | default | 未使用 |
| `mrope_interleaved` | true | 未使用（多模态 mRoPE） |
| `mrope_section` | [11, 11, 10] | 未使用（mRoPE 各维度段长，对应 t/h/w） |

> `partial_rotary_factor = 0.25` 表示只有每个 head 前 `256 × 0.25 = 64` 维参与 RoPE 旋转，
> 其余维度不加位置信息。`mrope_*` 是为多模态（时间/高/宽三维位置）准备的，纯文本推理不涉及。

## 三、文本塔权重清单（QwenWeights）

参与前向的文本权重（见 `QwenWeights`）：

- `embed_tokens`：词嵌入（因 `tie_word_embeddings=true`，同时用作 lm_head）；
- `layers[0..31]`：每层含 `input_norm` / `post_norm` / `mlp`，
  以及按类型二选一的 `lin`（linear attention）或 `full`（full attention）权重；
- `final_norm`：输出前最终 RMSNorm。

## 四、视觉塔（vision_config，当前未使用）

标准 ViT 编码器，用于把图像/视频 patch 编码到文本隐空间：

| 参数 | 值 |
| --- | --- |
| `depth` | 24（ViT block 数） |
| `hidden_size` | 1024 |
| `num_heads` | 16 |
| `intermediate_size` | 4096 |
| `hidden_act` | gelu_pytorch_tanh |
| `in_channels` | 3 |
| `patch_size` | 16 |
| `temporal_patch_size` | 2（视频时间维 patch） |
| `spatial_merge_size` | 2（空间 token 合并） |
| `num_position_embeddings` | 2304 |
| `out_hidden_size` | 2560（输出维度对齐文本塔 hidden_size） |

对应权重（见 `VisionWeights` / `VisionBlockWeights`）：patch embedding、位置 embedding、
24 个 block（norm1/norm2 + qkv/proj attention + fc1/fc2 MLP），以及把视觉特征投影到文本空间的 merger。

## 五、MTP 多 token 预测（当前未使用）

用于投机/并行解码加速：

| 参数 | 值 |
| --- | --- |
| `mtp_num_hidden_layers` | 1 |
| `mtp_use_dedicated_embeddings` | false（复用主 embedding） |

对应权重（见 `MtpWeights`）：`fc` / `norm` / `pre_fc_norm_embedding` / `pre_fc_norm_hidden`
及 1 个与主 Decoder 结构相同的 MTP 层。

## 六、多模态相关 token id（顶层配置）

| token | id | 说明 |
| --- | --- | --- |
| `image_token_id` | 248056 | 图像占位 token |
| `video_token_id` | 248057 | 视频占位 token |
| `vision_start_token_id` | 248053 | 视觉片段起始 |
| `vision_end_token_id` | 248054 | 视觉片段结束 |

纯文本推理不会用到以上 token。

## 七、与本项目实现的对应关系

- 配置解析：[`QwenConfig.cpp`](../src/llm/model/qwen/QwenConfig.cpp) 把上述字段解析到 `Data`/`TextConfig`/`VisionConfig`；
  标注"当前未使用"的字段仅解析保存，不参与前向。
- 权重加载：[`QwenWeights.h`](../src/llm/model/qwen/QwenWeights.h) 通过 mmap 零拷贝加载 safetensors，
  按层类型解析成 `LayerWeights`；视觉塔与 MTP 权重被解析但不进入推理。
- 层类型判定：实际前向按 `layer_types` 逐层区分 linear/full attention，而非用 `full_attention_interval` 推算。
- 请求状态：[`QwenSession.h`](../src/llm/model/qwen/QwenSession.h)，包含 full attention KV cache 与 linear attention recurrent state；
  scratch、已生成 token（`outputs`）、`max_seq_len` 上提到公共基类 [`SessionBase.h`](../src/backend/cuda/mem/SessionBase.h)。
- 整体前向：[`QwenModel.cpp`](../src/llm/model/qwen/QwenModel.cpp)（prefill / decode 前向路径）。
- 输出头：[`LMHead`](../src/llm/module/common/LMHead.h)（`common::LMHead`，Qwen / DeepSeek 共用）。
