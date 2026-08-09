# Qwen 前向 Module 设计

本文描述 [`src/llm/qwen/model/`](../src/llm/qwen/model) 下各 Module 的职责与相互关系。
Module 划分参照 [`qwen_architecture.md`](qwen_architecture.md) 的模型结构，写法上类比 PyTorch 的
`nn.Module`（每个 Module = 一段权重 + 一个 `forward`），但因本项目是手写 CUDA 推理，
在“谁持有什么”上与 PyTorch 有本质区别（见下）。

## 一、三条铁律（与 PyTorch 的关键区别）

所有 Module 都遵守以下约定，基类 [`Module`](../src/llm/qwen/model/Module.h) 的注释也写明了这三点：

1. **Module 不拥有权重**：权重是 mmap 的 `WeightData` 引用（见 [`QwenWeights.h`](../src/llm/qwen/QwenWeights.h)），
   device 副本由 `CudaWeightPool` 惰性上传并缓存。Module 只持 `const *Weights&` 引用 + `CudaWeightPool*`。
2. **Module 不持有临时激活**：前向中反复覆盖的中间 buffer 全部在 `QwenForwardScratch`
   （见 [`QwenForwardScratch.h`](../src/llm/qwen/QwenForwardScratch.h)），作为参数传入 `forward`。
3. **Module 不持有跨 token 状态**：KV cache / recurrent state 在 `QwenSession`
   （见 [`QwenSession.h`](../src/llm/qwen/QwenSession.h)），按 `layer_index` 取用。

推论：**Module 是无 per-request 状态的纯计算单元，天然并发安全**。
所有随请求变化的状态都在传入的 `QwenSession` / `QwenForwardScratch` 里，每请求一份，互不干扰。

三类内存的归属一目了然：

| 生命周期 | 内容 | 归属 |
| --- | --- | --- |
| 全程持久（跨请求） | 模型权重的 device 副本 | `CudaWeightPool` |
| 跨 token（一次请求内） | KV cache / recurrent + conv state | `QwenSession` |
| 反复覆盖（一次前向内） | 各层临时激活 | `QwenForwardScratch` |

## 二、Module 层次（组合关系）

```
QwenModel                        文本塔顶层（text_config 整体）
├── Embedding                    embed_tokens：token id -> hidden
├── DecoderLayer × 32            对应 LayerWeights，按 type 分派 attn 子层
│   ├── RMSNorm  (input_norm)    注意力子层前 pre-norm
│   ├── attn（二选一）
│   │   ├── FullAttention        FullAttnWeights：GQA + QK-Norm + partial RoPE + gate
│   │   └── LinearAttention      LinearAttnWeights：conv1d + gated delta / SSM
│   ├── RMSNorm  (post_norm)     MLP 子层前 pre-norm
│   └── SwiGLUMlp                MlpWeights：down(SiLU(gate(x)) * up(x))
├── RMSNorm      (final_norm)    塔尾归一化
└── LMHead                       复用 embed_tokens（tie）-> logits -> argmax
```

- `QwenModel` **拥有（组合）** 一个 `Embedding`、32 个 `DecoderLayer`、一个 `final_norm`（`RMSNorm`）、一个 `LMHead`。
- `DecoderLayer` **拥有** 两个 `RMSNorm`（input/post）、一个 `SwiGLUMlp`，以及一个 attn 子层——
  用基类指针 `std::unique_ptr<Module>` 持有，运行时按层类型是 `FullAttention` 或 `LinearAttention`。
- 层号 0..31 中，第 3、7、11、…、31 层（每第 4 层）为 `full_attention`，共 8 层；其余 24 层为 `linear_attention`
  （见架构文档 2.2）。实际判定按 `LayerWeights::type`，不靠 `full_attention_interval` 推算。

## 三、数据流（prefill / decode 两条路径）

每个 Module 的前向分两条路径，因为批量与单步的 kernel、KV 读写方式不同：

- **prefill**：一次处理整段 prompt（`tokens` 个位置），批量 kernel，写满各层 KV cache / recurrent state。
- **decode**：处理单个新 token（第 `pos` 位），单步 kernel，基于已有状态递推。

### prefill 流

```
h_input_ids (host, [tokens])
   │  Embedding.forward
   ▼
d_hidden [tokens, hidden]
   │  for layer in 0..31:  DecoderLayer.prefill(d_hidden, tokens, session, scratch)
   │     h  = x + attn( input_norm(x) )      // attn 读写 session 的 KV/state
   │     y  = h + mlp ( post_norm(h) )        // mlp 用 scratch
   ▼
d_hidden [tokens, hidden]
   │  final_norm（对末位）→ LMHead.forward（末位 [1, hidden]）
   ▼
下一个 token id
```

### decode 流（每步一个 token，循环至 EOS 或达上限）

```
prev_token_id (int)
   │  Embedding.forward（单 token）
   ▼
d_hidden [1, hidden]
   │  for layer in 0..31:  DecoderLayer.decode(d_hidden, pos, session, scratch)
   │     attn 追加写第 pos 位 KV / 递推 recurrent state（读自 session）
   ▼
d_hidden [1, hidden]
   │  final_norm → LMHead.forward
   ▼
下一个 token id
```

`QwenModel::prefill` 返回首个生成 token；`QwenModel::decode` 每次返回下一个 token。
外层循环（在 `QwenLLM` / 调用方）负责把 token 追加到 `session.h_outputs` 并推进 `pos`。

## 四、各 Module 与权重 / 状态的对应

| Module | 对应权重（QwenWeights.h） | 用到的跨 token 状态（QwenSession） | 主要 scratch 字段（QwenForwardScratch） |
| --- | --- | --- | --- |
| `Embedding` | `embed_tokens` | — | — |
| `RMSNorm` | 单个 norm `WeightData` | — | `norm_lowp_buffer` / `post_norm_lowp_buffer` |
| `FullAttention` | `FullAttnWeights` | `FullAttnKVCache` | `full_projection/q/gate/k/v/attn/attn_lowp` |
| `LinearAttention` | `LinearAttnWeights` | `LinearAttnRecurrentState` | `linear_projection/z/b/a/conv_out/gated/gated_lowp` |
| `SwiGLUMlp` | `MlpWeights` | — | `gate/up/gate_up/prod/prod_lowp/mixer/mlp_out_buffer` |
| `DecoderLayer` | `LayerWeights` | 按 layer_index 取上面二者之一 | `layer_out_buffer` / `token_hidden_a/b` |
| `LMHead` | `embed_tokens`（tie） | — | `y_buffer` / `argmax_*` |
| `QwenModel` | `QwenWeights` 整体 | 调度全部 | 调度全部 |

## 五、接口约定（CPU / GPU 变量区分）

- 激活流用 device float 裸指针，带 `d_` 前缀（如 `d_hidden`、`d_out`）。
- token id 用 host `std::vector<int>`（如 `h_input_ids`）或 `int`（单个），带 `h_` 前缀。
- 权重传 `const *Weights&` 引用（host mmap 引用），device 副本经 `CudaWeightPool` 取。
- 形状约定：`prefill` 的 rows = `tokens`，`decode` 的 rows = 1；`RMSNorm` / `SwiGLUMlp` 用 `rows` 统一两条路径。

## 六、当前不实现的部分

架构文档中的**视觉塔（VisionWeights）**与 **MTP（MtpWeights）** 已被 `QwenWeights` 解析但不参与前向，
因此 `model/` 下**不建**对应 Module（`ViTBlock` / `Merger` / `MtpLayer` 等）。等真正需要多模态或投机解码时再补，避免过度设计。

## 七、建议实现顺序

每步可独立验证，从简到难：

1. `RMSNorm` + `Embedding`（跑通 token -> hidden）
2. `SwiGLUMlp`（独立可测）
3. `FullAttention`（8 层，标准 GQA，参考项目有对照）
4. `LinearAttention`（最复杂，conv + SSM 递归，最后攻）
5. `DecoderLayer`（组合上面）→ `QwenModel` + `LMHead`（串全链）
