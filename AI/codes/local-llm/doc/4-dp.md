# DeepSeek-V2-Lite 推理正确性修复与性能优化（对齐 llama.cpp）

本文记录 local-llm 的 **DeepSeek-V2-Lite-Chat（Q4_K_M）** 推理路径从「能跑但输出乱码」到「输出与 llama.cpp 对齐」的正确性修复，以及在**同一张 RTX 3080** 上参考 Qwen 路径已有优化、逐步对齐 llama.cpp 的 decode 性能优化过程。

- 模型：`/home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf`（Q4_K_M，约 9.65 GiB，15.7B 参数，MLA + DeepSeekMoE）
- 硬件：RTX 3080（12GB，sm_86），CUDA 12
- 架构详情见 [`deepseek_architecture.md`](deepseek_architecture.md)

## 一、正确性修复

修复三处 dtype 阻塞（embedding / rms_norm / norm 权重 F32 读取）后引擎能运行，但 greedy 输出是乱码（`是是是` / `肺肺！` / `!!!!`）。通过逐项比对 211 上 llama.cpp 源码（`src/models/deepseek2.cpp`、`ggml/src/ggml-cpu/ops.cpp`、`src/llama-model.cpp`），定位并修复了两处数值 bug：

### 1. RoPE 类型：NeoX 半分 → GGML_ROPE_TYPE_NORM 相邻对

`llama-model.cpp` 中 `LLM_ARCH_DEEPSEEK2` 返回 `LLAMA_ROPE_TYPE_NORM`（"operating on pairs of consecutive head values"），即旋转**相邻对** `(vec[2i], vec[2i+1])`，pair `i` 用 `inv_freq[i]`；而原实现用的是 NeoX 半分 `(vec[i], vec[i+half])`。

修复 [`kernel.cu`](../src/backend/cuda/ops/kernel.cu) 的 `mla_rope_inplace`：

```cuda
// DeepSeek-V2 GGUF 采用 GGML_ROPE_TYPE_NORM（相邻对 interleaved），
// 即旋转 (vec[2i], vec[2i+1])，pair i 使用 inv_freq[i]，而非 NeoX 的半分。
const int half = rope_dim / 2;
for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float angle = static_cast<float>(pos) * inv_freq[i];
    const float c = cosf(angle), s = sinf(angle);
    const float x0 = vec[2 * i], x1 = vec[2 * i + 1];
    vec[2 * i]     = x0 * c - x1 * s;
    vec[2 * i + 1] = x0 * s + x1 * c;
}
```

该项修复后输出从乱码变为连贯英文，但语义仍偏离（"The term ... can"），说明还有次要数值偏差。

### 2. attn softmax scale：补上 YARN 的 mscale² 因子

llama.cpp deepseek2 的 `kq_scale = mscale² / sqrt(n_embd_head_k)`，其中：

```
attn_factor_org = attn_factor * (1 + 0.1 * ln(1/freq_scale))
mscale          = attn_factor_org * (1 + 0.1 * rope_yarn_log_mul * ln(1/freq_scale))
```

关键点：RoPE 的 cos/sin 幅值 mscale 在 `ext_factor != 0` 时净为 1.0（`attn_factor * (1 + 0.1*ln(40)) = 0.73053 * 1.36889 = 1.0`，见 `ops.cpp` 的 `rope_yarn`），所以 **RoPE 不做额外幅值缩放是对的**；但这也意味着 `attn_factor_org = 1.0`，因此 kq_scale 用的 `mscale = 1 + 0.1 * log_mul * ln(scale)`。

- `log_mul = 0.0707`，`ln(40) = 3.6889` → `mscale ≈ 1.02608`，`mscale² ≈ 1.0528`
- 正确 `kq_scale = 1.0528 / sqrt(192) = 0.07598`；原实现用 `1/sqrt(192) = 0.07217`（少了 5.3%）

修复 [`DeepseekSession.cpp`](../src/llm/model/deepseek/DeepseekSession.cpp)：

```cpp
float mscale = 1.0f;
if (config.use_yarn) {
    const float log_mul  = config.yarn_mscale;         // = yarn_log_multiplier
    const float ln_scale = std::log(config.yarn_scaling_factor);
    mscale = 1.0f + 0.1f * log_mul * ln_scale;
}
attn_softmax_scale = mscale * mscale / std::sqrt((float)config.qk_head_dim());
```

### 修复效果

两项修复后，greedy 输出与 llama.cpp ground truth 完全一致：

```
prompt:  User: What is the capital of France?\n\nAssistant:
输出：    The capital of France is Paris.
```

其他 prompt（如续写 "Once upon a time..."）也输出连贯英文（"girl named Cinderella. She was a kind and gentle girl who lived with her cruel stepmother..."）。至此正确性对齐完成。

## 二、性能基线（同一张 RTX 3080，同一 Q4_K_M GGUF）

| 实现 | 口径 | pp（prompt） | tg（decode） | 相对 llama.cpp（tg） |
| --- | --- | ---: | ---: | ---: |
| llama.cpp | `llama-bench -ngl 99 -fa 1 -p 32 -n 128 -r 3` | 1234.97 t/s | **132.68 t/s** | 1.0x |
| local-llm（修复后基线） | `--max-output-tokens 128` 墙钟 | — | **≈ 3.66 t/s** | **≈ 0.028x（慢约 36x）** |

> llama-bench 是纯稳态生成口径；local-llm 为 warmup 后墙钟 decode 平均。两者均全量 offload 到 GPU。

decode 差距约 **36 倍**，与预期一致：DeepSeek 路径尚未做 Qwen 路径已有的关键优化，当前瓶颈集中在 **MoE 逐 token / 逐专家的小 GEMM + host 回读**、缺少 CUDA Graph、以及低精度输入未复用等。

### 瓶颈定位（`--profile` 采样）

对 32 token decode 采样（`--profile-sample-every 8`）后，各 GPU kernel 耗时占比极小，所有已埋点 kernel 合计仅约 **10%** 的 decode 墙钟：

| kernel | count | total_ms | pct |
| --- | ---: | ---: | ---: |
| dequantize_q4k_to_f16 | 4035 | 83.4 | 0.42% |
| dequantize_q50/q80/q6k | 1645 | 103 | 0.51% |
| ds.gemm.egate/eup/edown | 8424 | 65.2 | 0.32% |
| f32_to_f16_copy | 9374 | 32.0 | 0.16% |
| silu_mul / moe_accumulate | 5751 | 18.7 | 0.09% |
| MLA / router / norm 等 | — | ~25 | ~0.12% |

而 `decode_token` 平均 281 ms/token，**约 90% 的时间不在任何 GPU kernel 里**，而是消耗在 host 侧：

- **MoE 路由 host 回读**：`MoERouter` 在 device topk 后 `to_host` 回读 expert_ids/weights，强制 device 同步、打断流水线。
- **逐 token 逐专家 eager 编排**：`RoutedExperts` 对每 token × 6 专家各发 3 个小 GEMM（decode 时约 500 个 kernel/token），全部走 host 循环逐个 launch，launch 开销主导。
- 侧证：SM util 仅 66%、mem-bw util 仅 8%，GPU 明显“喂不饱”，是被大量串行小 kernel 的 launch/同步拖住，而非算力或带宽瓶颈。
- 专家权重每 token 都在被重新反量化（4035 次 dequant / 32 token ≈ 126 次/token），因 F16 dequant pool LRU 抖动，但单次很便宜，聚合影响远小于 launch 开销。

**结论**：最高杠杆是 **P1（CUDA Graph 消除 per-kernel launch 开销）**，但它依赖 **P0（路由结果留 device、去掉 host 回读同步）**。故优化顺序 P0 → P1 优先。

### 运行命令

```bash
# local-llm（清理 GPU 残留进程后跑，双缓存池调低避免 OOM）
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
printf -v P 'User: What is the capital of France?\n\nAssistant:'
LOCAL_LLM_CUDA_WEIGHT_POOL_GB=6 LOCAL_LLM_CUDA_DEQUANT_POOL_GB=3 \
  PROMPT="$P" ./build-dsopt/local_llm \
  --model deepseek --model-dir /home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
  --max-output-tokens 128

# llama.cpp
./build-cuda/bin/llama-bench -m /home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 32 -n 128 -r 3
```

## 三、优化计划与进展

参考 Qwen 路径已落地的优化（见 [`4.md`](4.md) / [`4-2.md`](4-2.md)），对 DeepSeek 路径按优先级推进：

- **P0**（进行中）：MoE router 结果留在 device 端不回读；RoutedExperts 分组 / 批量专家 GEMM，消除逐 token 逐专家的小 GEMM 循环。
- **P1**：DeepseekModel.decode 引入 CUDA Graph + device 端 argmax 闭环（依赖 P0）。
- **P2**：MLA / DenseFFN / SharedExperts / MoE 复用 `prepare_lowp_input` + `gemm_lowp`，同一 hidden 只转一次低精度输入。
- **P3**：`add_rms_norm` 融合（残差加 + RMSNorm 合并 kernel）。

> 每档优化均在 211 上实测，与 llama.cpp（tg=132.68 t/s）对照记录，逐步更新下表。

| 阶段 | 关键改动 | tg (t/s) | 相对基线 | 相对 llama.cpp |
| --- | --- | ---: | ---: | ---: |
| 基线（修复后） | 正确性对齐 | 3.66 | 1.0x | 0.028x |
| P0（权重留 device） | decode 时 MoE top_w 不回读 host，加权累加直接读 device 权重 | 3.70 | ~1.0x | 0.028x |

### P0 实测与结论：单点消同步不足以撬动 decode

尝试了两版 P0：

1. **单核融合 MoE decode kernel**（把每层 6 专家 × (gate/up GEMM + silu + down GEMM + 累加) ≈ 30 次 launch 融成 1 个 kernel）。结果长生成退化为乱码且不提速，两个根因：
   - 正确性：融合路径收集 k 个专家反量化 F16 切片指针后即释放 LRU lease，长生成时 27 层 × 多 token churn 触发 dequant pool 驱逐，异步 kernel 仍在读被驱逐/复用的 buffer → 间歇性数据竞争（16-token 短跑不触发，故当时 "Paris" 正确）。
   - 性能：把 3k 个 device 权重指针传给 kernel 每层要一次 `cuda_memcpy_h2d`（内含 `cudaStreamSynchronize`），等于把每层 2 次 `to_host` 换成 1 次指针上传同步，无净收益。

2. **务实回退版**（已保留）：decode 时 MoE 的 top_w 权重留在 device，不再 `to_host`，加权累加从 device 直接读权重，去掉每层一次同步；prefill 路径不变。复用已验证正确的 cuBLAS GEMM 循环，lease 生命周期安全。

- 正确性：`--max-output-tokens 16` 严格输出 "The capital of France is Paris."；128 token 连贯英文无乱码。
- 性能：**3.66 → 3.70 t/s（噪声范围内，≈持平）**。

**关键结论**：单点消除某一次 host 同步不足以撬动 decode。真正瓶颈是**每层约 30 次 kernel launch + 仍保留的每层 1 次索引 `to_host` 同步**，二者叠加在 27 层上放大。要拿到 profiling 预期的大幅提升，必须上 **P1 CUDA Graph**：把整段 decode 捕获成图、一次 replay 消除所有 per-kernel launch 开销。但 CUDA Graph 要求 decode 全程零 host 回读，这又反过来要求：
- MLA 三个 kernel（`mla_rope_q` / `mla_kv_a` / `mla_attend`）改为读 device 端 `pos`（对齐 Qwen 的 `FullAttention.decode`）；kv_b gather / GEMM 的随步增长 `seq` 要按 Qwen 已有做法处理。
- MoE 路由的 k 个 int 索引也不能回读 host——但专家权重是量化的、需按 host 索引反量化，这是与 CUDA Graph 冲突的根本约束。可选解：专家权重预取/常驻（受 12GB 显存限制，需分层或量化直算）、或持久化 dequant lease + 单次无同步指针传递。
- lm_head 改为 device 端 argmax，把下一 token 写回 device buffer 形成闭环（对齐 Qwen 的 `forward_argmax_device`）。

这是一次跨 MLA/MoE/Session/Model 的较大重构，风险与工作量都显著高于 P0。

### P1 可行性调研：F16-dequant 架构与 12GB 显存的根本冲突

深入调研 CUDA Graph 前置条件后，发现一个**架构级根本约束**，决定了在本卡上单靠 CUDA Graph 无法追平 llama.cpp：

1. **CUDA Graph 已具备的条件**（对齐 Qwen 路径即可）：`FullAttention.decode` 读 `int *d_pos`（device 端 pos）、`LMHead::forward_argmax_device`（GPU argmax 写回 device token，可直接复用）、`CudaGraph` capture/replay API、KV cache 已是固定 `max_seq_len` 尺寸。MLA 三个 kernel 改读 device pos、lm_head 走 device argmax 都可行。
2. **真正的死结在 MoE 权重**：本引擎的 GEMM 路径是「量化权重 → 按 host 已知的专家索引反量化成 F16 → cuBLAS」。CUDA Graph replay 时不能按 host 索引做反量化（图内 kernel 参数必须固定），故路由必须留 device；但留 device 又无法驱动 host 侧的按索引反量化——**根本冲突**。
3. **显存不够做 F16 常驻**：64 专家 × 3 tensor × (1408×2048 F16=5.77MB) × 26 层 ≈ **28.8GB**，远超 12GB，无法把专家权重以 F16 常驻。
4. **量化常驻 + F16 dequant 也塞不下**：实测 `WEIGHT_POOL_GB=10 DEQUANT_POOL_GB=1.5` 直接 OOM（`cudaMalloc ...ffn_gate_exps.weight.e63.dequant.f16 失败`）。量化全模型 9.65GB + F16 dequant 缓存 + KV/scratch 在 12GB 上无法共存；且 `CudaWeightPool` 是 **evict-all** 策略，6GB 池装不下 9.65GB 全量权重，必然反复驱逐重传（profile 里 dequantize_q4k 被调 4035 次即此故）。

**根因**：llama.cpp 之所以能在同卡跑到 132 t/s，是因为它**直接做量化 GEMV（不展开成 F16）**，把 9.65GB 量化模型整体常驻显存、还留有余量；而本引擎走「反量化成 F16 再 cuBLAS」，在 12GB 卡上既放不下 F16、量化常驻又与 CUDA Graph 的 device 索引冲突。

**结论**：要真正追平 llama.cpp，需实现**量化 GEMV kernel（Q4_K / Q5_0 / Q6_K / Q8_0 直算，权重量化常驻、device 索引驱动 MoE）**，这等价于重写 ggml 的 CUDA MoE 路径。**本轮已按此结论落地**（见下一节），从根本上消除了 F16-dequant 与显存/同步的冲突。

## 四、量化直算重构：消除 F16-dequant，量化权重整体常驻

针对 P1 调研定位的根本约束，重构了 GEMM 与 Embedding 的权重访问路径，从「量化权重 → 反量化成 F16 → cuBLAS」改为**「量化权重常驻 device + 量化直算 kernel（on-the-fly 解块）」**，与 llama.cpp 的量化直算路径一致。

### 1. 量化直算 GEMM kernel（`quant_gemv_kernel`）

在 [`kernel.cu`](../src/backend/cuda/ops/kernel.cu) 新增一族模板 kernel，支持 `Y[M,out] = X[M,in]·W[out,in]^T`，权重按 GGUF 原始量化格式常驻，kernel 内**逐元素 on-the-fly 反量化**，不再展开成 F16：

- 每个 warp 负责一个输出行 `o`，32 lane 沿 `in_dim` 分工点积后 warp 归约；内层循环 M 个 token（decode M=1、prefill M>1 通用）。
- 逐 dtype 提供 `q4k_at / q6k_at / q50_at / q80_at` 解块函数，块布局与对应 `dequantize_*` kernel 严格一致（Q4_K 144B/256、Q6_K 210B/256、Q5_0 22B/32、Q8_0 34B/32），保证数值等价。
- 入口 [`TensorTool::gemm`](../src/tensor/TensorTool.cpp) 对**所有量化权重**统一走此路径（M 任意），彻底不再触发 F16 dequant。

### 2. 量化直算 Embedding（`quant_embedding_kernel`）

`token_embd`（Q4_K，`[102400,2048]`，F16 展开约 0.42GB）原先经 `to_gpu(true)` 整表反量化。新增按 token id **只反量化命中行**的 kernel，消除这最后一处大 F16 lease。

### 3. 权重整体常驻

去掉 F16 dequant lease 后，量化全模型（约 9.65GB）可整体常驻在 `CudaWeightPool`（默认 10GB 上限即可容纳，不再触发 evict-all 抖动重传），加上 KV cache（`max_seq_len × 576 × 4B × 27层`，仅数十 MB）、scratch 与 CUDA context，稳定落在 12GB 内。

### 4. 实测效果（RTX 3080，同一 Q4_K_M GGUF）

| 阶段 | tg (t/s) | 相对基线 | 说明 |
| --- | --- | --- | --- |
| 修复正确性后基线 | ~3.7 | 1x | 反量化成 F16 → cuBLAS + evict-all 抖动 |
| 量化直算 GEMV（权重部分常驻） | ~22.5 | ~6x | 短 prompt 实测，正确性保持（Paris） |
| 量化直算 + 量化权重整体常驻 | ~42.8 | ~11.6x | 无 evict-all 抖动；正确性保持 |

> 正确性：`What is the capital of France?` → `The capital of France is Paris.`，与 llama.cpp ground truth 一致。

与 llama.cpp 的 132.68 t/s 相比仍有差距，剩余差距主要来自 host launch/同步开销（每层约 30 次 kernel launch + 路由回读），需 **CUDA Graph** 收口——量化直算已扫清其前置的 F16-dequant/显存障碍（量化权重常驻、可按 device 索引选块），是后续可选的收尾项。

### 已落地成果小结

- **正确性**：DeepSeek-V2-Lite 输出已与 llama.cpp 完全对齐（修复 RoPE 类型 + attn softmax mscale²）。
- **量化直算重构**：GEMM/Embedding 改为量化权重常驻 + on-the-fly 解块，消除 F16-dequant 与 evict-all 抖动。
- **性能**：tg 从 ~3.7 → **~42.8 t/s（~11.6x）**，正确性保持；与 llama.cpp 132.68 t/s 的剩余差距待 CUDA Graph 收口。
