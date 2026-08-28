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
| llama.cpp | `llama-bench -ngl -1 -p 10 -n 128 -r 3` | 715.26 ± 210.15 t/s | **232.33 ± 0.69 t/s** | 1.0x |
| local-llm（最新默认：correctness-safe） | prompt 实测 10 tokens，`--max-output-tokens 128` | — | **≈ 4.0 t/s** | **≈ 0.017x（慢约 58x）** |
| local-llm（显式实验：`edown+sdown` 直通） | 同 prompt，`LOCAL_LLM_DEEPSEEK_QUANT_GEMV_OPS=edown,sdown` | — | **≈ 4.89 t/s** | **≈ 0.021x（慢约 47.5x）** |
| local-llm（修复后基线） | `--max-output-tokens 128` 墙钟 | — | **≈ 3.66 t/s** | **≈ 0.016x（慢约 63.5x）** |

> llama-bench 是纯稳态生成口径；local-llm 为 warmup 后墙钟 decode 平均。两者均全量 offload 到 GPU。

最新 correctness-safe 默认路径与 llama.cpp 的 decode 差距约 **58 倍**；显式打开 `edown/sdown` 直通可到约 **47.5 倍**，但不作为默认正确性/稳定性路径。DeepSeek 路径尚未做 llama.cpp 的 device-indexed MoE 量化直算与高效调度，当前瓶颈集中在 **MoE 逐 token / 逐专家的小 GEMM + host 回读**、缺少 CUDA Graph、以及大量小 kernel launch。

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

### 最新等条件 profile（nsys / ncu / 内置 profiler）

同一台 211、同一张 RTX 3080、同一 `DeepSeek-V2-Lite-Chat-Q4_K_M.gguf`，采用短跑口径 `p=10 / tg=32`：

| 实现 | 命令口径 | tg |
| --- | --- | ---: |
| llama.cpp | `llama-bench -m ... -p 10 -n 32 -ngl -1 -r 3` | **233.37 ± 1.83 t/s** |
| local-llm | prompt 实测 10 tokens，`--max-output-tokens 32`，correctness-safe 默认路径 | **≈ 4.0–4.6 t/s** |

采集情况：

- `nsys profile` 可以生成原始 trace：`/tmp/local-llm-prof/nsys_local_tg32.qdstrm`（437 MiB）与 `/tmp/local-llm-prof/nsys_llamacpp_tg32.qdstrm`（467 MiB）。
- 211 当前 Nsight Systems 缺 importer 依赖，`nsys profile --stats=true` 无法在远端导出统计表：`Importer error status: The importer binary and its dependencies were not found`。
- `ncu` 受性能计数器权限限制，`basic` / `LaunchStats` 都被 `ERR_NVGPUCTRPERM` 拦截，当前用户无法采集硬件 counter。
- 因此本轮可用的细分数据来自 local-llm 内置低扰动 profiler：`/tmp/local-llm-profile-latest/profile_deepseek_summary.md`。

local-llm 最新 profile 关键结论：

- 32 token decode：`wall_ms=6292.5`，`tokens_per_sec=5.085`，平均 `196.6 ms/token`。
- 设备利用率：SM util 平均约 `64%`，mem-bw util 平均约 `7.7%`，说明不是显存带宽打满，而是 GPU 被 host 调度/同步喂不饱。
- GPU kernel 单项耗时很小：`ds.gemm.egate/eup` 平均约 `0.007–0.008 ms`，`ds.gemm.edown` 约 `0.004 ms`，`ds.gemm.sdown` 约 `0.003 ms`。
- 仍有大量 safe path 反量化：`dequantize_q4k_to_f16` 5608 次、累计约 `106.6 ms`；但相对 6.3s decode 墙钟仍不是主因。
- 失败实验：尝试把 decode route ids 的 D2H 改为 pinned host + copy stream 异步拷贝，并提前计算 shared experts 来覆盖等待；exact-output 发生偏移且吞吐降到约 `4.73 t/s`，已回退。原因是 shared/routed 权重访问与 dequant lease 的时序被重排，当前 eager 路径下不能简单通过跨 stream overlap 保证数值/生命周期等价。
- 失败实验：尝试把单个 routed expert 的 `edown quant GEMV + moe_accumulate_device` 合成一个 per-route kernel，保持 host 侧 `r=0..5` 顺序不变；exact-output 仍偏移且吞吐约 `4.71 t/s`，已回退。说明即使只少一个 accumulate launch，去掉 `edown` 中间结果落全局内存也会改变 greedy token 路径，不能作为默认优化。
- trace 工具：新增 `LOCAL_LLM_DEEPSEEK_TRACE=1` 与 `LOCAL_LLM_DEEPSEEK_TRACE_POS=<pos>`，可输出每层 `embedding/mla/mlp/final_norm` 的 hidden fingerprint。safe vs `edown` / `sdown` 对比显示首次可见差异都出现在第一个 MoE 层 `pos=18 layer=1 stage=mlp`，单步差异很小，但长文本 greedy 会逐步放大。
- 正确性修复：safe dequant path 中 `dequantize_to_f16` 和 `f32_to_f16_copy` 必须与后续 cuBLAS GEMM 使用同一个 `get_current_cuda_stream()`；之前混用 `nullptr/default stream` 与当前 stream，存在跨流未排序风险。
- 失败实验：`CudaWeightPool` 曾尝试按 LRU 逐项淘汰，64 token profile 中权重上传从 `42782.8 MiB` 降到 `25430.6 MiB`，H2D 从 `4712.3 ms` 降到 `2887.7 ms`，吞吐 `4.02 → 4.69 t/s`；但 128 token 连续运行出现 segfault。由于目标 GPU 默认内存够用，当前已移除 weight-pool LRU 能力，恢复“超限整体清空”的简单稳定策略。

由此可见，local-llm 与 llama.cpp 的主要差距不是单个 GEMV kernel 的算力，而是 **MoE 路由/专家选择仍在 host 侧编排、每 token 每层大量小 kernel launch、expert ids 回读同步、safe dequant lease 与权重池调度**。后续应优先把 MoE decode 改成 device-indexed 批量量化 GEMV，再谈 CUDA Graph 收口。

### 运行命令

```bash
# local-llm（清理 GPU 残留进程后跑，双缓存池调低避免 OOM）
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
printf -v P 'User: What is the capital of France?\n\nAssistant:'
LOCAL_LLM_CUDA_WEIGHT_POOL_GB=6 LOCAL_LLM_CUDA_DEQUANT_POOL_GB=2 \
  PROMPT="$P" ./build-dsopt/local_llm \
  --model deepseek --model-dir /home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
  --max-output-tokens 128

# llama.cpp
./build-cuda/bin/llama-bench -m /home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
  -ngl -1 -p 10 -n 128 -r 3
```

## 三、优化计划与进展

参考 Qwen 路径已落地的优化（见 [`4.md`](4.md) / [`4-2.md`](4-2.md)），对 DeepSeek 路径按优先级推进：

- **P0**（进行中）：MoE router 结果留在 device 端不回读；RoutedExperts 分组 / 批量专家 GEMM，消除逐 token 逐专家的小 GEMM 循环。
- **P1**：DeepseekModel.decode 引入 CUDA Graph + device 端 argmax 闭环（依赖 P0）。
- **P2**：MLA / DenseFFN / SharedExperts / MoE 复用 `prepare_lowp_input` + `gemm_lowp`，同一 hidden 只转一次低精度输入。
- **P3**：`add_rms_norm` 融合（残差加 + RMSNorm 合并 kernel）。

> 每档优化均在 211 上实测，与 llama.cpp 最新等条件口径（`p10/tg128`，tg=232.33 t/s）对照记录，逐步更新下表。

| 阶段 | 关键改动 | tg (t/s) | 相对基线 | 相对 llama.cpp |
| --- | --- | ---: | ---: | ---: |
| 基线（修复后） | 正确性对齐 | 3.66 | 1.0x | 0.016x |
| P0（权重留 device） | decode 时 MoE top_w 不回读 host，加权累加直接读 device 权重 | 3.70 | ~1.0x | 0.016x |
| 当前默认 | device argmax + routed expert view 预缓存 + correctness-safe dequant path | ~4.0 | ~1.09x | 0.017x |
| 显式实验 | `LOCAL_LLM_DEEPSEEK_QUANT_GEMV_OPS=edown,sdown` 选择性量化直通 | 4.89 | 1.34x | 0.021x |

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

**根因**：llama.cpp 最新等条件口径能在同卡跑到 232.33 t/s，是因为它**直接做量化 GEMV（不展开成 F16）**，把 9.65GB 量化模型整体常驻显存、还留有余量；而本引擎走「反量化成 F16 再 cuBLAS」，在 12GB 卡上既放不下 F16、量化常驻又与 CUDA Graph 的 device 索引冲突。

**结论**：要真正追平 llama.cpp，需实现**量化 GEMV kernel（Q4_K / Q5_0 / Q6_K / Q8_0 直算，权重量化常驻、device 索引驱动 MoE）**，这等价于重写 ggml 的 CUDA MoE 路径。但后续 exact-output 与 trace 对拍表明，全量/局部量化直算虽然单 GEMM 误差很小，跨层累计后仍会改变 greedy token，因此当前只能作为显式实验路径。

## 四、量化直算调研与第一阶段落地

针对 P1 调研定位的根本约束，重构了 GEMM 与 Embedding 的权重访问路径，探索从「量化权重 → 反量化成 F16 → cuBLAS」改为**「量化权重常驻 device + 量化直算 kernel（on-the-fly 解块）」**。但 DeepSeek 的 greedy 输出对逐层数值漂移非常敏感，后续 exact-output 与 trace 对拍发现直通会从第一层 MoE 开始引入微差，因此当前默认策略改为：**Embedding 保持量化查表；DeepSeek 层内 GEMM 默认使用 correctness-safe dequant+cuBLAS；量化直通只保留为显式实验 allowlist**。

### 1. 量化直算 GEMM kernel（`quant_gemv_kernel`）

在 [`kernel.cu`](../src/backend/cuda/ops/kernel.cu) 新增一族模板 kernel，支持 `Y[M,out] = X[M,in]·W[out,in]^T`，权重按 GGUF 原始量化格式常驻，kernel 内**逐元素 on-the-fly 反量化**，不再展开成 F16：

- 每个 warp 负责一个输出行 `o`，32 lane 沿 `in_dim` 分工点积后 warp 归约；内层循环 M 个 token（decode M=1、prefill M>1 通用）。
- 逐 dtype 提供 `q4k_at / q6k_at / q50_at / q80_at` 解块函数，块布局与对应 `dequantize_*` kernel 对齐（Q4_K 144B/256、Q6_K 210B/256、Q5_0 22B/32、Q8_0 34B/32）。
- 入口 [`TensorTool::gemm`](../src/tensor/TensorTool.cpp) 对量化权重支持直算；DeepSeek `ds.gemm.*` 默认仍使用 safe path，可用 `LOCAL_LLM_DEEPSEEK_QUANT_GEMV_OPS=edown,sdown,egate,eup` 逐项测试直通候选，也可用 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_QUANT_GEMV=1` 测试全量直通。

### 2. 量化直算 Embedding（`quant_embedding_kernel`）

`token_embd`（Q4_K，`[102400,2048]`，F16 展开约 0.42GB）原先经 `to_gpu(true)` 整表反量化。新增按 token id **只反量化命中行**的 kernel，消除这最后一处大 F16 lease。

### 3. 正确性边界与当前默认策略

单 GEMM 对拍显示量化直通与 safe dequant+cuBLAS 的误差通常只有 `1e-4 ~ 1e-3` 量级，但 DeepSeek 多层 greedy 会放大这些差异：

- `d_attn,kv_` 直通：输出会明显跑偏（出现 URL-like 文本）。
- `egate/eup/edown` 任意两个或三个同时直通：短输出会退化为重复 `!`。
- 单独 `egate`：当前长输出测试会触发异常退出；单独 `eup`：输出严重跑偏；`edown` 与 `sdown` 的 F32 operands 直通短跑可对齐，但长文本 exact-output 会间歇分叉，不能作为默认。
- F16 operands 实验：`LOCAL_LLM_EXPERIMENTAL_QUANT_GEMV_F16_OPERANDS=1` 会让 `edown` 实验路径输出偏移；`sdown` 单独 F16 operands 也会偏移。因此“on-the-fly 解量化后先 round 到 F16”并不等价于 safe path，当前不能作为扩大直通范围的默认策略。

因此当前不是“全量量化直通”，而是**correctness-safe 默认 + 显式直通实验**：默认保留 `ds.gemm.*` safe dequant+cuBLAS 主路径，量化直通只用于受控对拍和性能实验。

### 4. 实测效果（RTX 3080，同一 Q4_K_M GGUF）

| 阶段 | tg (t/s) | 相对基线 | 说明 |
| --- | ---: | ---: | --- |
| 修复正确性后基线 | ~3.66 | 1.0x | `ds.gemm.*` safe dequant+cuBLAS |
| P0（权重留 device） | ~3.70 | ~1.0x | MoE top_w 不回读 host |
| `ds.gemm.edown` 量化直通 + device argmax（显式实验） | **~4.60–4.68** | **~1.26x** | 短跑可对齐；长文本 trace 显示 layer=1 MLP 起出现微差 |
| Routed expert view 预缓存 | **~4.62–4.64** | **~1.27x** | 构造期预切 64 个 expert 的 gate/up/down view，减少 decode 热循环的 slice 与字符串构造 |
| `ds.gemm.edown + ds.gemm.sdown` 量化直通（显式实验） | **~4.85–5.06** | **~1.35x** | 性能较好，但长文本 exact-output 不稳定 |

> 正确性：`What is the capital of France?` → `The capital of France is Paris.`；Qwen/DeepSeek 功能测试改为 exact-output 断言，211 上 `ctest` 5/5 通过。

与 llama.cpp 最新等条件 `232.33 t/s` 相比仍有巨大差距。当前真正可行的后续方向不再是盲目扩大全量量化直通，而是按 exact-output 测试逐项推进：

### 已落地成果小结

- **正确性**：DeepSeek-V2-Lite 输出已与 llama.cpp 完全对齐（修复 RoPE 类型 + attn softmax mscale²）。
- **测试护栏**：Qwen 保留长输出 exact-output；DeepSeek 改为稳定事实型 exact-output，同时用 trace 工具分析长文本 greedy 分叉。
- **量化直通策略**：DeepSeek `ds.gemm.*` 默认 safe dequant+cuBLAS；`edown/sdown/egate/eup` 通过 `LOCAL_LLM_DEEPSEEK_QUANT_GEMV_OPS` 显式打开，全量直通和 F16 operands 保留为实验开关。
- **decode 尾部优化**：DeepSeek greedy decode 复用 Qwen 的 device embedding + GPU argmax，避免每步把完整 logits 回传 host，仅回传一个 token id，为后续 CUDA Graph 铺路。
- **host 热路径瘦身**：Routed experts 在构造期预缓存每个 expert 的 `StorageTensor` view，decode 时按 route id 直接取数组元素，避免每 token 反复 slice 和拼接名字。
- **性能**：当前 correctness-safe 默认 story prompt tg 约 **4.0 t/s**；显式 `edown/sdown` 可到 **4.85–5.06 t/s**，但不作为默认。剩余差距仍主要来自 MoE host 编排、expert id 回读、权重池/反量化缓存抖动和每层大量小 kernel launch。
- **失败实验**：尝试把 6 个 expert 的 `edown + accumulate` 合成单个 device-indexed 量化 kernel，但 exact-output 发生偏移且吞吐降到约 **2.9 t/s**。根因是单 warp 串行处理 6 个 expert 降低并行度，同时累加路径的细微数值差异仍会改变 greedy token，因此已回退。
- **已移除实验**：weight-pool LRU 曾减少权重重复上传，但 128 token 连续运行仍出现 segfault；当前 `CudaWeightPool` 已去掉 LRU 能力，默认依赖足够 GPU 内存，容量不足时仍整体清空。

### 后续可行优化

- **优先级 1：device-indexed MoE 量化 GEMV**。把 `expert_ids` 保持在 device，让 kernel 按 route id 直接定位量化 expert 权重，避免 host 回读和每 expert 逐个 launch。这是对齐 llama.cpp MoE decode 的关键。
- **优先级 2：融合 MoE gate/up/silu/down/accumulate 的 decode 专用 kernel**。上次 F16 指针融合失败的根因是 dequant lease 生命周期和指针上传同步；量化直算后可在 kernel 内用 expert id 直接解块读取，绕开 F16 lease。
- **优先级 3：DeepSeek CUDA Graph**。当前已具备 device token/argmax；还需要 MLA 读 device pos、MoE route 全 device 化，才能捕获整段 decode。
- **优先级 4：数值等价量化直算修复**。继续对拍 `egate/eup` 与 safe path，若能把多层漂移压到 exact-output 不变，可逐步扩大直通范围。
