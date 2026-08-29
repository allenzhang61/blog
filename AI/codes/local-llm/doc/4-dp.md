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
- 失败实验：`CudaWeightPool` 曾尝试按 LRU 逐项淘汰，64 token profile 中权重上传从 `42782.8 MiB` 降到 `25430.6 MiB`，H2D 从 `4712.3 ms` 降到 `2887.7 ms`，吞吐 `4.02 → 4.69 t/s`；但 128 token 连续运行出现 segfault。由于目标 GPU 默认内存够用，当前已移除 weight-pool LRU 能力，并删除超限自动清空逻辑；`CudaWeightPool` 不再设置人工容量上限，显存不足时由 `cudaMalloc` 报真实 OOM。

由此可见，local-llm 与 llama.cpp 的主要差距不是单个 GEMV kernel 的算力，而是 **MoE 路由/专家选择仍在 host 侧编排、每 token 每层大量小 kernel launch、expert ids 回读同步、safe dequant lease 与权重池调度**。后续应优先把 MoE decode 改成 device-indexed 批量量化 GEMV，再谈 CUDA Graph 收口。

### 运行命令

```bash
# local-llm（清理 GPU 残留进程后跑；CudaWeightPool 无人工上限）
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
printf -v P 'User: What is the capital of France?\n\nAssistant:'
LOCAL_LLM_CUDA_DEQUANT_POOL_GB=1 \
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
4. **量化常驻 + F16 dequant 也塞不下**：历史实测 `WEIGHT_POOL_GB=10 DEQUANT_POOL_GB=1.5` 直接 OOM（`cudaMalloc ...ffn_gate_exps.weight.e63.dequant.f16 失败`）。量化全模型 9.65GB + F16 dequant 缓存 + KV/scratch 在 12GB 上无法共存；当前 `CudaWeightPool` 不再自动清空，也不再设置人工容量上限，容量问题会直接表现为真实 `cudaMalloc` OOM。

**根因**：llama.cpp 最新等条件口径能在同卡跑到 232.33 t/s，是因为它**直接做量化 GEMV（不展开成 F16）**，把 9.65GB 量化模型整体常驻显存、还留有余量；而本引擎走「反量化成 F16 再 cuBLAS」，在 12GB 卡上既放不下 F16、量化常驻又与 CUDA Graph 的 device 索引冲突。

**结论**：要真正追平 llama.cpp，需实现**量化 GEMV kernel（Q4_K / Q5_0 / Q6_K / Q8_0 直算，权重量化常驻、device 索引驱动 MoE）**，这等价于重写 ggml 的 CUDA MoE 路径。但后续 exact-output 与 trace 对拍表明，全量/局部量化直算虽然单 GEMM 误差很小，跨层累计后仍会改变 greedy token，因此当前只能作为显式实验路径。

## 四、量化直算调研与第一阶段落地

针对 P1 调研定位的根本约束，重构了 GEMM 与 Embedding 的权重访问路径，探索从「量化权重 → 反量化成 F16 → cuBLAS」改为**「量化权重常驻 device + 量化直算 kernel（on-the-fly 解块）」**。但 DeepSeek 的 greedy 输出对逐层数值漂移非常敏感，后续 exact-output 与 trace 对拍发现直通会从第一层 MoE 开始引入微差，因此当前默认策略改为：**Embedding 保持量化查表；DeepSeek 层内 GEMM 默认使用 correctness-safe dequant+cuBLAS；量化直通只保留为显式实验 allowlist**。

### 1. 量化直算 GEMM kernel（`quant_gemv_kernel`）

在 [`kernel.cu`](../src/backend/cuda/ops/kernel.cu) 新增一族模板 kernel，支持 `Y[M,out] = X[M,in]·W[out,in]^T`，权重按 GGUF 原始量化格式常驻，kernel 内**逐元素 on-the-fly 反量化**，不再展开成 F16：

- 每个 warp 负责一个输出行 `o`，32 lane 沿 `in_dim` 分工点积后 warp 归约；内层循环 M 个 token（decode M=1、prefill M>1 通用）。
- 逐 dtype 提供 `q4k_at / q6k_at / q50_at / q80_at` 解块函数，块布局与对应 `dequantize_*` kernel 对齐（Q4_K 144B/256、Q6_K 210B/256、Q5_0 22B/32、Q8_0 34B/32）。
- 入口 [`TensorTool::gemm`](../src/tensor/TensorTool.cpp) 对量化权重支持直算；DeepSeek `ds.gemm.*` 默认仍使用 safe path，可用 `LOCAL_LLM_DEEPSEEK_QUANT_GEMV_OPS=edown,sdown,egate,eup` 逐项测试直通候选，也可用 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_QUANT_GEMV=1` 测试全量直通。
- 新增 llama.cpp-style 实验旁路：`LOCAL_LLM_EXPERIMENTAL_QUANT_GEMV_Q8_1=1` 时，已进入 quant GEMV 的 decode `m==1` 会先把 activation 动态量化为 Q8_1（每 32 元素 36B：`f16 d + f16 sum + int8 qs[32]`），再做 `quant weight × Q8_1 activation`。该开关只改变 quant GEMV 算法，不单独强制 DeepSeek 绕过 safe path；仍需配合 `LOCAL_LLM_DEEPSEEK_QUANT_GEMV_OPS=...` 或全量直通实验开关。当前 Q4_K/Q6_K 已实现 block-level dot，Q5_0/Q8_0 仍为逐元素 fallback。

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
| `edown+sdown` + Q8_1 activation 直通（显式实验） | **~4.75–4.76** | **~1.30x** | 128 token 可跑通；Q4_K/Q6_K block-level dot 已实现，但当前仍不如 F32 activation 直通 |
| 删除 weight-pool 自动清空和人工上限后 | — | — | `CudaWeightPool` 只受真实显存限制；若量化常驻 + dequant/runtime 超过 12GB，则直接 `cudaMalloc` OOM |
| 阶段 1：DeepSeek decode 全量量化直通（显式实验） | **~34.6–35.6 t/s** | **~9.5x** | 短问答可跑通；全量 `ds.gemm.*` decode 跳过 `try_dequant()`，但输出数值基线已改变，长 story 会提前 EOS 或受真实 OOM 限制 |
| 阶段 2：MoE fused gate/up/SiLU（显式实验） | **~36.8–37.8 t/s** | **~10.1x** | 在阶段 1 基础上打开 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1`，decode 的 routed/shared experts 用 fused `gate+up+SiLU` quant kernel；`edown/sdown` 仍走阶段 1 quant GEMV |
| 阶段 3：prefill quant matmul（显式实验） | **~40–41 t/s** | **~11x** | 对齐 llama.cpp 的核心数据流：prefill 先把 activation 动态量化成 Q8_1，再用 Q8_1 activation + quant weight 做 matmul；修复 MLA latent gather 跨流 D2D copy 后，128-token story 可无 trace 稳定续写 |
| 阶段 3 full tiled MMQ 实验（quant-direct 模式） | **~48.3 t/s** | — | `LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1` 会自动启用该路径；使用 llama-style `block_q8_1_mmq` group-major packed activation、fixed `I=128,J=8,K=256`、padded `tile_x` shared-memory staging 与 DP4A 内层；正确性优先 |
| decode Q8_1 覆盖补齐（quant-direct 模式） | **~48.8–48.9 t/s** | — | quant-direct 模式下除 `ds.gemm.*` 外，`lm_head` 也走 Q8_1 GEMV；该模式不再需要额外的 strict no-dequant 验证开关 |
| Packed expert GPU 常驻（默认） | **~48.8–49.5 t/s** | **~13.5x** | MoE packed expert tensor 整块上传，`.eX` expert 只做 GPU pointer view；weight allocation 从几千个降到 GGUF tensor 级别，128 token story 不再 OOM |
| decode device-indexed MoE（quant-direct 模式） | **~53.7–54.5 t/s** | **~14.7x** | quant-direct 模式下 decode `top_idx/top_w` 留在 device；routed experts 用 `ds.gemm.e_indexed_moe` 两段 kernel 处理 6 个 route，减少 host route 回读和 per-expert launch |
| MLA `kv_b` expanded cache（quant-direct 模式） | **~57.4–57.7 t/s** | **~15.7x** | 每层新增 `[max_seq_len, n_heads*(qk_nope+v_head)]` F32 cache；prefill 写入整段，decode 只投影当前 token 的 normalized latent，避免每步重算全历史 `kv_b`；12/12 单测通过 |
| routed down route-parallel（quant-direct 模式） | **~60.0–60.2 t/s** | **~16.4x** | `quant_down_q8_1_indexed_accum` 从一 warp 串行遍历 6 个 route 改为一 warp 处理一个 `(route, hidden)`，用 `atomicAdd` 累加；ncu 单 kernel 约 `125us -> 65us`，12/12 单测通过 |

> 正确性：阶段 1/2 的 France smoke 均可生成 `The capital of France is Paris.`；阶段 3 改为 llama.cpp-style Q8_1 activation 后短问答能生成“巴黎/法国首都巴黎”一类语义正确回答；修复 MLA latent gather 的 stream 顺序后，story 长文 smoke 可无 trace 连贯续写。但 DeepSeek 仍不能通过英文 exact-output，事实回答表述/语言仍会分叉，不能把 quant-direct 路径作为默认正确性路径。

与 llama.cpp 最新等条件 `232.33 t/s` 相比仍有巨大差距。当前真正可行的后续方向不再是盲目扩大全量量化直通，而是按 exact-output 测试逐项推进：

### 5. 移除 `try_dequant()` 的阶段任务清单

目标是把 DeepSeek 从当前的 `quant weight -> dequantize_to_f16 -> cuBLAS` 逐步切换为 `quant weight -> quant direct kernel`，让 12GB RTX 3080 上尽量做到量化权重常驻，不再依赖大块 F16 dequant cache。这个切换会改变数值轨迹，因此需要把 quant-direct 作为新的实验基线，而不是继续要求与 safe dequant+cuBLAS 完全逐 token 相等。

- **阶段 1：decode GEMV 全覆盖**。新增 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_DECODE_QUANT_DIRECT=1`，仅在 `ds.gemm.*` 且 `m == 1` 时跳过 `try_dequant()`，所有 DeepSeek decode 量化 GEMM 走 `launch_quant_gemv` / `launch_quant_gemv_q8_1`；prefill `m > 1` 仍保留 safe path。
- **阶段 2：MoE 专项优化**。面向 `egate/eup/edown/sgate/sup/sdown` 做 shape-specialized quant kernel，减少 routed/shared experts 的小 GEMM launch 和中间写回。
- **阶段 3：prefill quant matmul**。补齐 `m > 1` 的 quant MMQ/quant matmul，避免长 prompt 仍依赖 `try_dequant()`。
- **阶段 4：切换默认路径**。在 decode、prefill、质量 smoke、长文本连续运行都稳定后，再把 DeepSeek 默认切到 quant-direct，并保留 `LOCAL_LLM_FORCE_SAFE_DEQUANT_GEMM=1` 作为回退开关。

阶段 1 的覆盖范围：

- MLA：`ds.gemm.d_attn_q`、`ds.gemm.kv_a`、`ds.gemm.kv_b`、`ds.gemm.d_attn_output`
- Dense FFN：`ds.gemm.d_ffn_gate`、`ds.gemm.d_ffn_up`、`ds.gemm.d_ffn_down`
- MoE router：`ds.gemm.router`
- Routed experts：`ds.gemm.egate`、`ds.gemm.eup`、`ds.gemm.edown`
- Shared experts：`ds.gemm.sgate`、`ds.gemm.sup`、`ds.gemm.sdown`

阶段 1 初测：

- 命令：`LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_DECODE_QUANT_DIRECT=1 LOCAL_LLM_CUDA_DEQUANT_POOL_GB=0.05`，France prompt，`--max-output-tokens 16 --profile --profile-sample-every 4`。
- 结果：生成 `The capital of France is Paris.`，实际 decode 7 tokens，stdout 吞吐约 `34.6 t/s`，profile 汇总约 `35.6 t/s`。
- 显存：采样峰值 `device used ≈ 9038 MiB`，`CudaWeightPool` 驻留峰值 `≈ 7048 MiB`，H2D 上传总量 `≈ 7048 MiB`，没有 weight-pool 驱逐。
- 限制：长 story prompt 在阶段 1 下可能 prefill 后直接 EOS；128 token 场景在 12GB RTX 3080 上仍可能触发真实 `cudaMalloc` OOM，因为阶段 1 只移除 decode 的 F16 dequant，prefill/warmup 仍保留 safe dequant，且量化权重常驻会持续增长。

阶段 2 初测：

- 实现：新增 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1`，仅在 `m == 1`、gate/up 量化 dtype 与 shape 一致时接管 MoE decode；否则自动 fallback 到原 `egate/eup/sgate/sup + silu_mul`。
- 覆盖：routed experts 的 `egate+eup+silu_mul` 合并为 `ds.gemm.e_swiglu`，shared experts 的 `sgate+sup+silu_mul` 合并为 `ds.gemm.s_swiglu`；`edown/sdown` 仍沿用阶段 1 的 quant GEMV。
- 命令：`LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_DECODE_QUANT_DIRECT=1 LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU=1 LOCAL_LLM_CUDA_DEQUANT_POOL_GB=0.05`，France prompt，`--max-output-tokens 16 --profile --profile-sample-every 4`。
- 结果：生成 `The capital of France is Paris.`，实际 decode 7 tokens，stdout 吞吐约 `36.8–37.5 t/s`，profile 汇总约 `37.8 t/s`。
- 命中：profile 中 `ds.gemm.e_swiglu` 2496 次、`ds.gemm.s_swiglu` 52 次；对应的 decode routed/shared gate/up/SiLU 已由 fused kernel 接管。
- 显存：延迟分配 fallback-only gate/up scratch 后，采样峰值 `device used ≈ 9008 MiB`，`CudaWeightPool` 驻留峰值 `≈ 7050 MiB`，H2D 上传总量 `≈ 7050 MiB`，与阶段 1 基本同量级。

阶段 3 初测：

- 实现：新增 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_QUANT_DIRECT=1`，仅在 `ds.gemm.* && m > 1` 时跳过 `try_dequant()`；初版 `launch_quant_matmul` 是 float activation + on-the-fly weight dequant，每个 warp 负责一个 `(token,row)` 输出元素。
- llama.cpp 对齐修正：llama.cpp 的 CUDA MMQ 不是 float activation 逐元素直乘，而是先把 activation 动态量化成 Q8_1，再用 quant weight × Q8_1 activation 做 block dot / tiled MMQ。local-llm 已新增 `launch_quant_matmul_q8_1`，阶段 3 现在先 `launch_quantize_q8_1`，再走 Q8_1 prefill matmul；旧 float `launch_quant_matmul` 保留为 fallback/单测路径。
- 覆盖：prefill 中的 MLA、dense FFN、router、shared experts 等 `m > 1` GEMM 可走 quant matmul；routed experts 当前代码仍逐 token 调用，`m == 1` 时依赖阶段 1 decode quant-direct 开关。
- 单测：新增 `LaunchQuantMatmulTest.ComputesQ80MultipleRows` 和 `LaunchQuantMatmulQ81Test.ComputesQ80MultipleRows`，分别验证 float activation matmul 与 Q8_1 activation matmul；211 上两者均通过。
- 命令：推荐使用转正后的 `LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1`，等价于对 DeepSeek `ds.gemm.*` 自动打开 decode quant-direct、decode Q8_1 activation、prefill Q8_1 quant matmul 和 MoE fused SwiGLU；旧的 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_Q8_1_QUANT_DIRECT_PRESET=1` 暂保留兼容。仍可用 `LOCAL_LLM_DEEPSEEK_DECODE_QUANT_DIRECT_EXCLUDE_OPS` / `LOCAL_LLM_DEEPSEEK_PREFILL_QUANT_DIRECT_EXCLUDE_OPS` 把单个 op 回退 safe path。
- 结果：程序可跑完，实际 decode 16 tokens，stdout 吞吐约 `41.3 t/s`；France prompt 输出 `巴黎，法国的首都是巴黎...` 或同义事实回答，不再是无关英文续写，但仍不等于 exact-output 期望 `The capital of France is Paris.`。
- 显存：packed expert 修复后，阶段 3 France prompt 末值 `weight_allocs=377`，`CudaWeightPool` 驻留约 `9880 MiB`，`device used≈10382 MiB`；没有 F16 dequant 大缓存依赖。
- 长跑：packed expert 修复后 128 token story prompt 可跑完且不 OOM；在修复 MLA latent gather 的 stream 顺序问题后，无 trace quant-direct 也可稳定续写英文 story，stdout 吞吐约 `21.2 t/s`。
- 质量 preset：新增脚本 `scripts/deepseek_quant_quality_ablate.sh`。preset 单独运行可避免 OOM/segfault 和多数低层乱码，但仍会跑题；64-token story 常见输出变为中文新闻/英文续写，而不是 robot painting story。
- Ablation：单独回退 `ds.gemm.router`、`ds.gemm.edown`、`ds.gemm.sdown`、`ds.gemm.d_attn_output`、`ds.gemm.kv_b` 都没有稳定恢复 prompt 语义；`decode Q8_1` 能明显缓解空行/`!` 循环，但无法解决语义跑偏。
- 逐层对拍：新增 `LOCAL_LLM_DEEPSEEK_TRACE=1`、`LOCAL_LLM_DEEPSEEK_TRACE_TAG`、`LOCAL_LLM_DEEPSEEK_TRACE_POS`、`LOCAL_LLM_DEEPSEEK_TRACE_ROW`，并输出 `embedding/attn_normed/attn_q_rope/kv_a/kv_latent/kv_b_out/attn_ctx/attn_out/mla/moe_normed/router_logits/router_topk/moe_out/moe_post_add/mlp/final_norm` 的统计摘要；`scripts/deepseek_trace_compare.py` 可比较两份 trace。France prompt 的 pos=14 对拍显示 safe1/safe2 完全一致，只开 prefill Q8_1 时第一次 top-k 分叉在 layer7，preset 全开时第一次 top-k 分叉在 layer9；将 prefill 所有 `ds.gemm.*` 回 safe 后可恢复一致，说明当前阶段3 prefill Q8_1 仍是语义漂移的主要风险源，保留为显式实验路径。
- llama.cpp 对拍：在 llama.cpp 中新增 env-gated DeepSeek trace（`LLAMA_DEEPSEEK_TRACE`、`LLAMA_DEEPSEEK_TRACE_POS`、`LLAMA_DEEPSEEK_TRACE_STAGES`、`LLAMA_DEEPSEEK_TRACE_LAYER`、`LLAMA_DEEPSEEK_TRACE_ROW`），用 `llama-simple` 跑同一 raw prompt `法国的首都是` 时 token 数为 4、trace pos=3，生成 `巴黎`。local safe 的 `mlp` 与 llama.cpp 的 `l_out` 高层输出整体接近，后半层相对 RMS 多在约 1% 或更低；MoE 语义映射应使用 `ffn_norm -> moe_normed`、`ffn_moe_logits -> router_logits`、`ffn_out -> moe_out`（因为 `ffn_out = ffn_moe_out + ffn_shexp`）、`l_out -> mlp`。router top-k 对比显示 local safe 与 llama.cpp 不是 bit-level 对齐，26 个 MoE 层中 9 层 top-k 集合不一致、23 层顺序不一致；local quant-direct 与 llama.cpp 统计相近（10 层集合不一致、24 层顺序不一致），没有证明 quant-direct 比 safe 明显更偏离 llama.cpp。
- Q8_1 `sum` 对齐：llama.cpp 的 CUDA decode/MMVQ `vec_dot_q4_K_q8_1_impl_vmmq` 并不使用 `block_q8_1.ds.y`，而是在 dot 内对 q8 `qs` 求和后乘 `d8`；prefill/MMQ 的 `vec_dot_q4_K_q8_1_impl_mmq` 才使用 `ds8.y` 的 raw `sum(x)`。local 新增 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_Q8_1_RAW_SUM=1` 做隔离 A/B：只在 prefill Q8_1 quantize 写 raw `sum(x)`，decode 仍保留 `d * sum(qs)`。France pos=3 safe/quant 对拍显示 raw-sum prefill 反而更差（top-k mismatch `12 -> 16`，large tensor diffs `86 -> 294`，logits `first_diff 0.122 -> 0.340`、`rms_rel 0.0069 -> 0.0115`），因此不作为默认；当前默认继续使用 `d * sum(qs)`。补充单测已覆盖 Q4_K × Q8_1 的 scale/min compensation 与 Q6_K × Q8_1 的 scale group，证明 local block dot 公式本身自洽。后续若要更贴近 llama.cpp MMQ，应整体实现 `block_q8_1_mmq` 144B 打包、`I=128/J=8..128/K=256` tiled MMQ 和对应累加顺序，而不是单独改 Q8_1 header。
- Full tiled MMQ 实验：`q8_1_mmq` activation 已改成 llama.cpp kernel 期望的 group-major 布局 `[group][token][block_q8_1_mmq]`，而不是 local 早期实验的 `[token][group]`。`launch_quant_matmul_q8_1_mmq()` 现在使用 fixed `I=128,J=8,K=256`：每个 CUDA block 覆盖 128 个输出行和 8 个 token，8 个 warp 分别对应 8 个 token，shared memory staging 当前 K-superblock 的两个 128-wide Q8_1 activation group，以及 Q4_K/Q6_K 的 padded `tile_x`。Q4_K 使用 `x_qs[i*(32+1)+...]`、`x_dm[i]`、`x_sc[i*4+i/8+...]` 三段；Q6_K 使用 `x_qs[i*(64+1)+...]`、`x_df[i]`、`x_sc[i*4+i/8+...]` 三段，内层用 `__dp4a` 做 4-way int8 dot。按“llama.cpp 正确性优先”原则，tiled MMQ 使用 raw `sum(x)` 作为 Q4_K min compensation，并已接入 `LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1`。France pos=3 与 llama.cpp mapped trace 对比：full tile + DP4A 的 logits `first_diff=0.738`、`rms_rel=3.28%`、`sum_diff=19827`，比早期 tiled raw-sum 的 `first_diff=0.780`、`rms_rel=3.67%` 有所改善；64-token story 可稳定输出，当前 smoke 吞吐约 `48.3 t/s`。当前仍只固定 `J=8`，尚未实现 llama.cpp 的动态 `J=8..128` 选择、全部模板分发和 stream-k/fixup。
- decode Q8_1 覆盖补齐：`LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1` 覆盖 DeepSeek 层内 `ds.gemm.*` 的 decode Q8_1 GEMV，同时把通用 op 名 `"lm_head"` 纳入 Q8_1 GEMV，更接近 llama.cpp decode/MMVQ 的 quant weight × Q8_1 activation 数据流。旧的 `LOCAL_LLM_STRICT_NO_DEQUANT=1` 验证开关已删除；quant-direct 模式下是否触发 `try_dequant()` 由分发逻辑本身保证，普通 correctness-safe 默认仍允许 safe dequant。211 上 stable quant-direct 入口的 France profile 可见 `lm_head` 命中，输出 `巴黎，还是法国...`，16-token profile 约 `54.6 t/s`。
- decode device-indexed MoE：新增 `LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_DEVICE_INDEXED_MOE=1`，并由 `LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1` 自动启用。decode `input_size == 1` 时 `MoERouter` 不再把 `top_idx` 回读到 host，而是把 `top_idx/top_w` 的 device 指针传给 `RoutedExperts`；`RoutedExperts` 优先调用 `TensorTool::moe_routed_decode_indexed()`。该路径分两段：先用 `launch_quant_swiglu_indexed()` 按 device expert id 一次生成 6 个 route 的 `[k, expert_ffn]` act，再把 act 量化成 Q8_1，并用 `launch_quant_down_q8_1_indexed_accum()` 一次完成 routed down 与按 route 权重累加。下投影权重实测为 `Q8_0`，因此 indexed down 已补齐 `Q4_K/Q6_K/Q5_0/Q8_0` 分发；输出行内串行遍历 route，避免 atomic 和非确定累加顺序。211 上 `LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1` 的 profile 中 `ds.gemm.e_indexed_moe` 为 52 次（2 次采样 × 26 个 MoE 层），decode routed expert 已接管；仍存在的 `ds.gemm.e_swiglu/edown` 624 次来自 4-token prefill 的 `4 × 26 × 6`，prefill routed experts 后续再单独处理。
- Stream 修复：story prompt 曾出现“无 trace quant-direct 输出中文重复，但 `LOCAL_LLM_DEEPSEEK_TRACE=1` 或 `CUDA_LAUNCH_BLOCKING=1` 正常”的现象。定位后发现 MLA 的 `ds.gather.latent` 使用同步 `cudaMemcpy2D`，没有排入当前 non-blocking compute stream，可能在 KV-cache 写入 kernel 完成前从默认流读取旧数据；改为 `cudaMemcpy2DAsync(..., get_current_cuda_stream())` 后，16-token 与 128-token story 在无 trace 下均恢复正常。`MoE::forward` 的 `g_moe_out` 清零也改为当前流上的 `cudaMemsetAsync`，避免后续 routed/shared expert 累加与默认流清零存在类似顺序隐患。
- 旧 OOM：在 packed expert GPU 常驻前，128 token story prompt 仍会 OOM；不带 profile 时阶段 2 在 `blk.17.ffn_up_exps.weight.e31` OOM，阶段 3 在 `blk.10.ffn_gate_exps.weight.e48` OOM。后续 profile 证明瓶颈不是 KV/scratch，而是每个 expert 单独 `cudaMalloc` 带来的 allocator/page 开销。
- 测试：211 上 `local_llm` / `local_llm_tests` 构建通过；`ctest` 为 12/12；新增 `LaunchQuantizeQ81Test.SupportsRawAndQuantizedBlockSums` 覆盖 Q8_1 header 的 `d * sum(qs)` 与 raw `sum(x)` 两种写法，并新增 Q4_K/Q6_K × Q8_1 focused 单测锁住 K-quant block dot 公式；DeepSeek story smoke 在单独打开 `LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=1` 下可跑完并输出连贯英文故事。

Packed expert GPU 常驻：

- 定位：新增 `CudaAllocTracker` 后，DeepSeek 阶段 2 短跑的 `weight_allocs` 约 `3569`，`untracked device used` 约 `2.0 GiB`；Qwen 同口径只有 `426` 个 weight allocation、`untracked` 约 `378 MiB`。
- 验证：独立脚本按 GGUF packed tensor 整块 `cudaMalloc` 时，量化权重 `9866.98 MiB` 可放下，allocator overhead 约 `235 MiB`；模拟把 expert tensor 按 64 份拆开时，`allocs=4925`、overhead 约 `2406 MiB` 并 OOM。
- 实现：`StorageTensor::slice` 记录底层 storage name/data/shape/nbytes/byte offset；`CudaWeightPool::cached_weight()` 遇到 slice 时先上传 packed base tensor，再缓存无所有权的 `base.ptr + offset` view，view 不增加 `cached_bytes()`，也不产生 H2D。
- 防护：`moe_router_topk_kernel` 对 NaN/Inf logits 做 sanitize，避免输出 `-1` expert；`RoutedExperts` 对 host 回读 expert id 做边界检查，避免非法 id 变成越界访问。
- 结果：阶段 2 France prompt 输出 `The capital of France is Paris.`，decode 7 tokens，stdout 吞吐约 `48.8 t/s`；`weight_allocs` 降到 `377`，`untracked` 降到约 `500 MiB`。
- 长跑：128 token story prompt 不带 profile 可跑完，stdout 吞吐约 `49.5 t/s`，不再 OOM；带 profile 也可跑完，峰值 `CudaWeightPool` 驻留约 `9880 MiB`，`tracked_alloc` 约 `9942 MiB`，末值 `device used` 约 `10438 MiB`，`untracked` 约 `496 MiB`。
- 限制：quant-direct 输出仍会发散，例如 story 中出现重复 `!`/中英混杂；这解决的是显存/allocator OOM 和吞吐，不代表阶段 4 默认切换条件已满足。

### 已落地成果小结

- **正确性**：DeepSeek-V2-Lite 的 RoPE 类型与 attn softmax mscale² 已修复，但 greedy exact-output 仍会因微小数值差异出现 token 分叉，不能只靠单条事实问答判定默认正确性。
- **测试护栏**：Qwen 保留长输出 exact-output；DeepSeek exact-output 当前仍不稳定，继续依赖 smoke/profile/trace 组合定位分叉点。
- **量化直通策略**：DeepSeek `ds.gemm.*` 默认 safe dequant+cuBLAS；decode 全量直通、MoE fused SwiGLU、device-indexed routed MoE、prefill quant matmul、F16 operands、Q8_1 activation 直通均保留为显式实验开关。
- **decode 尾部优化**：DeepSeek greedy decode 复用 Qwen 的 device embedding + GPU argmax，避免每步把完整 logits 回传 host，仅回传一个 token id，为后续 CUDA Graph 铺路。
- **host 热路径瘦身**：Routed experts 在构造期预缓存每个 expert 的 `StorageTensor` view，decode 时按 route id 直接取数组元素，避免每 token 反复 slice 和拼接名字。
- **显存定位**：DeepSeek MoE packed expert tensor 不能按 expert 拆成几千个 GPU allocation；packed base 常驻 + expert pointer view 后，`untracked` 从约 **2.0 GiB** 降到约 **500 MiB**，128 token story OOM 消失。
- **性能**：当前 correctness-safe 默认 story prompt tg 约 **4.0 t/s**；阶段 1/2 短问答实验可到 **~35–38 t/s**，packed expert GPU 常驻后阶段 2 路径约 **49 t/s**；decode device-indexed MoE 后短跑约 **54 t/s**。阶段 3 已消除 prefill safe dequant 依赖，修复 stream 顺序后 128-token story smoke 可连贯输出但仍非 exact-output，暂不作为默认。剩余差距主要来自 prefill routed experts、未做 CUDA Graph，以及量化直通数值轨迹仍未完全贴近 llama.cpp。
- **失败实验**：尝试把 6 个 expert 的 `edown + accumulate` 合成单个 device-indexed 量化 kernel，但 exact-output 发生偏移且吞吐降到约 **2.9 t/s**。根因是单 warp 串行处理 6 个 expert 降低并行度，同时累加路径的细微数值差异仍会改变 greedy token，因此已回退。
- **已移除实验**：weight-pool LRU 曾减少权重重复上传，但 128 token 连续运行仍出现 segfault；当前 `CudaWeightPool` 已去掉 LRU 能力和人工容量上限，默认依赖足够 GPU 内存，显存不足时由 `cudaMalloc` 直接报 OOM，不再整体清空。

### 后续可行优化

- **优先级 1：device-indexed MoE 量化 GEMV**。把 `expert_ids` 保持在 device，让 kernel 按 route id 直接定位量化 expert 权重，避免 host 回读和每 expert 逐个 launch。这是对齐 llama.cpp MoE decode 的关键。
- **优先级 2：融合 MoE gate/up/silu/down/accumulate 的 decode 专用 kernel**。上次 F16 指针融合失败的根因是 dequant lease 生命周期和指针上传同步；量化直算后可在 kernel 内用 expert id 直接解块读取，绕开 F16 lease。
- **优先级 3：DeepSeek CUDA Graph**。当前已具备 device token/argmax；还需要 MLA 读 device pos、MoE route 全 device 化，才能捕获整段 decode。
- **优先级 4：数值等价量化直算修复**。继续对拍 `egate/eup` 与 safe path，若能把多层漂移压到 exact-output 不变，可逐步扩大直通范围。
