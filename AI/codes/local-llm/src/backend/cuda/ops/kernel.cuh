//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_KERNEL_CUH
#define LOCAL_LLM_KERNEL_CUH

#include <cstddef>
#include <cstdint>

#include "tensor/TensorCommon.h"

// 手写 CUDA kernel 的 launch 接口。矩阵乘不在此处（走 cuBLAS，见 gemm.h），
// 这里只放 cuBLAS 覆盖不到的逐元素 / 归一化 / 采样等算子。
//
// stream 用 void* 传递以避免让 .cpp 侧被 nvcc 编译：实现内部按 cudaStream_t 使用，
// nullptr 表示默认流。
//
// 低精度类型约定：lowp_type 0 = bfloat16，1 = float16（与权重 dtype 对应）。

// ---- 逐元素 ----
// 残差加：out = a + b，n 个元素（out 可与 a 或 b 相同做原位）。
void launch_add(const float *a, const float *b, float *out, int n, void *stream);

// 手写 bf16 GEMV：Y[out_dim] = W[out_dim, in_dim] · X[in_dim]（decode 单 token 投影专用）。
// W 行主序 bf16、X bf16、Y f32。in_dim 需为偶数（Qwen 全部满足）。
// 针对 M=1（GEMV，访存瓶颈）替代 cuBLAS：cuBLAS 在 M=1 会选 tensor-core GEMM tile，
// 64×64 tile 仅 1 行有效，算力大量浪费；手写 GEMV 直接按访存带宽跑满。
void launch_bf16_gemv(const uint16_t *weight, const uint16_t *x, float *y,
                      int out_dim, int in_dim, void *stream);

// 量化直算 GEMM：Y[M,out_dim] = X[M,in_dim] · W[out_dim,in_dim]^T，权重量化常驻、on-the-fly 反量化。
// quant_type 指明权重量化格式；row_bytes 为每行量化字节数。X f32，Y f32，均 row-major。
// f16_operands=true 时，on-the-fly 解出的权重和输入先 round 到 F16，再转回 F32 累加，用于贴近 safe path。
// 避免把量化权重展开成 F16（省显存 + 省一次读写），直接在 kernel 内解块，追平 llama.cpp 的量化直算路径。
void launch_quant_gemv(DType quant_type, const uint8_t *weight, size_t row_bytes, const float *x,
                       float *y, int out_dim, int in_dim, int m, bool f16_operands, void *stream);

// 量化直算 MATMUL：每个 warp 负责一个 (token,row) 输出元素，面向 prefill m>1。
// 相比 launch_quant_gemv 的 token 维串行循环，这里按 m*out_dim 并行，避免 prefill 仍依赖 F16 dequant cache。
void launch_quant_matmul(DType quant_type, const uint8_t *weight, size_t row_bytes, const float *x,
                         float *y, int out_dim, int in_dim, int m, bool f16_operands, void *stream);

// llama.cpp-style 实验路径：先把 activation 动态量化成 Q8_1（每 32 个元素 36 字节），
// 再用量化权重与 Q8_1 activation 做 GEMV/MMQ。
size_t q8_1_row_bytes(int in_dim);
void launch_quantize_q8_1(const float *x, uint8_t *x_q8_1, int in_dim, int m, void *stream,
                          bool store_raw_sum = false);
void launch_quant_gemv_q8_1(DType quant_type, const uint8_t *weight, size_t row_bytes,
                            const uint8_t *x_q8_1, float *y,
                            int out_dim, int in_dim, void *stream);
void launch_quant_matmul_q8_1(DType quant_type, const uint8_t *weight, size_t row_bytes,
                              const uint8_t *x_q8_1, float *y,
                              int out_dim, int in_dim, int m, void *stream);
size_t q8_1_mmq_row_bytes(int in_dim);
void launch_quantize_q8_1_mmq(const float *x, uint8_t *x_q8_1, int in_dim, int m, void *stream,
                              bool store_raw_sum = false);
void launch_quant_matmul_q8_1_mmq(DType quant_type, const uint8_t *weight, size_t row_bytes,
                                  const uint8_t *x_q8_1, float *y,
                                  int out_dim, int in_dim, int m, void *stream);

// DeepSeek MoE decode 专用：同时计算 gate/up 两个量化 GEMV，并直接写出
// SiLU(gate) * up，减少 egate + eup + silu_mul 三次 launch 和中间张量写回。
void launch_quant_swiglu(DType quant_type, const uint8_t *gate_weight, const uint8_t *up_weight,
                         size_t gate_row_bytes, size_t up_row_bytes, const float *x, float *act,
                         int ffn_dim, int in_dim, bool f16_operands, void *stream);
void launch_quant_swiglu_indexed(DType quant_type, const uint8_t *gate_weight, const uint8_t *up_weight,
                                 size_t gate_expert_bytes, size_t up_expert_bytes,
                                 size_t gate_row_bytes, size_t up_row_bytes, const float *x,
                                 const int *expert_ids, float *act, int k, int ffn_dim,
                                 int in_dim, bool f16_operands, void *stream);
void launch_quant_down_q8_1_indexed_accum(DType quant_type, const uint8_t *down_weight,
                                          size_t down_expert_bytes, size_t down_row_bytes,
                                          const uint8_t *act_q8_1, const int *expert_ids,
                                          const float *route_weights, float *out,
                                          int k, int hidden_size, int ffn_dim, void *stream);

// SwiGLU 门控：out = SiLU(gate) * up，n 个元素。
void launch_silu_mul(const float *gate, const float *up, float *out, int n, void *stream);

// ---- Embedding ----
// 批量查表：按 token_ids 从低精度权重表 [vocab, hidden] 取行转 float 到 output[tokens, hidden]。
void launch_embedding_lookup(const int *input, float *output, const uint16_t *table,
                             int input_size, int vocab_size, int hidden_size, int weight_type, void *stream);

// 量化直算查表：table 量化常驻（每行 row_bytes 字节），按 token id 只反量化命中行到 f32，
// 避免把整张 [vocab,hidden] 量化表展开成 F16。quant_type 指明权重量化格式。
void launch_quant_embedding(DType quant_type, const int *input, float *output, const uint8_t *table,
                            size_t row_bytes, int hidden_size, int input_size, void *stream);

// ---- RMSNorm ----
// weight_type 指明 norm 权重（gamma）的 dtype：0=bf16，1=f16。
//
// RMSNorm 输出 float：对每行 [hidden] 归一化后乘以 weight，写回 output[rows, hidden]。
// one_plus 为 true 时使用 (1 + weight) 作为缩放（部分 Qwen norm 的约定）。
void launch_rms_norm(const float *input, float *output, const uint16_t *weight, int weight_type,
                     int rows, int hidden_size, float eps, bool one_plus, void *stream);
// 融合 add + RMSNorm：out_residual = x + residual；out_norm = rmsnorm(out_residual) * weight。
void launch_add_rms_norm(const float *x, const float *residual, float *out_residual, float *out_norm,
                         const uint16_t *weight, int weight_type, int rows, int hidden_size,
                         float eps, bool one_plus, void *stream);

// ================= full attention =================
// q_norm / k_norm 权重按 (1 + w) 缩放（Qwen 约定），以 bf16 传入（q_norm_weight[head_dim]）。
// q_and_gate 布局：每个 query head 连续 [head_dim(q), head_dim(gate)]，故长度 tokens * n_heads * head_dim * 2。
// 输出 q / gate 拆分为 [tokens, n_heads * head_dim]。RoPE 仅作用前 head_dim*partial_rotary_factor 维。

// 单 token：处理位置 pos（pos 为 device 端 int，供 CUDA Graph 复用时逐步更新）。
void launch_full_attention_q(const float *q_and_gate, const uint16_t *q_norm_weight,
                             float *q, float *gate,
                             int n_heads, int head_dim, const int *pos_dev,
                             float rope_theta, float partial_rotary_factor, float eps,
                             void *stream);

// 批量：start_pos 为本段第一个 token 的绝对位置。
void launch_full_attention_q_batch(const float *q_and_gate, const uint16_t *q_norm_weight,
                                   float *q, float *gate,
                                   int tokens, int n_heads, int head_dim, int start_pos,
                                   float rope_theta, float partial_rotary_factor, float eps,
                                   void *stream);

// k/v：归一化 + RoPE(k) 后写入 KV cache（cache 布局 [max_seq_len, kv_heads * head_dim]）。
// kv_bf16=true 时 key_cache/value_cache 为 __nv_bfloat16 存储，否则 float。
void launch_full_attention_kv(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                              void *key_cache, void *value_cache, bool kv_bf16,
                              int kv_heads, int head_dim, int max_seq_len, const int *pos_dev,
                              float rope_theta, float partial_rotary_factor, float eps,
                              void *stream);

void launch_full_attention_kv_batch(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                                    void *key_cache, void *value_cache, bool kv_bf16,
                                    int tokens, int kv_heads, int head_dim, int max_seq_len, int start_pos,
                                    float rope_theta, float partial_rotary_factor, float eps,
                                    void *stream);

// causal attention + 输出门控（out = softmax(qk/sqrt(d)) · v * sigmoid(gate)）。
void launch_full_attention_attend(const float *q, const float *gate,
                                  const void *key_cache, const void *value_cache, bool kv_bf16, float *attn,
                                  int n_heads, int kv_heads, int head_dim, int max_seq_len,
                                  const int *pos_dev,
                                  void *stream);

void launch_full_attention_attend_batch(const float *q, const float *gate,
                                        const void *key_cache, const void *value_cache, bool kv_bf16, float *attn,
                                        int tokens, int n_heads, int kv_heads, int head_dim,
                                        int max_seq_len, int start_pos, void *stream);

// ================= linear attention =================
// depthwise 因果卷积（kernel=4）+ SiLU，conv_weight bf16，布局 [conv_dim, kernel]。
// conv_state 为跨 token 的滑动窗口，布局 [conv_dim, kernel]。
void launch_linear_attention_conv(const float *mixed, const uint16_t *conv_weight,
                                  float *conv_state, float *conv_out,
                                  int conv_dim, int kernel, void *stream);

void launch_linear_attention_conv_batch(const float *mixed, const uint16_t *conv_weight,
                                        float *conv_state, float *conv_out,
                                        int tokens, int conv_dim, int kernel, void *stream);

// gated delta 递归：更新 recurrent_state 并输出 gated（value_total）。
// a_log 为 float [value_heads]，dt_bias 为 bf16 [value_heads]，norm_weight 为 float [v_dim]。
void launch_linear_attention_recurrent(const float *conv_out, const float *z, const float *b,
                                       const float *a, const float *a_log, const uint16_t *dt_bias,
                                       const float *norm_weight, void *recurrent_state, bool state_bf16,
                                       float *gated, int key_heads, int value_heads, int k_dim, int v_dim,
                                       float eps, void *stream);

void launch_linear_attention_recurrent_batch(const float *conv_out, const float *z, const float *b,
                                             const float *a, const float *a_log, const uint16_t *dt_bias,
                                             const float *norm_weight, float *recurrent_state, float *gated,
                                             int tokens, int key_heads, int value_heads, int k_dim, int v_dim,
                                             float eps, void *stream);

// 融合 kernel：conv1d + gated delta 递归 + 读出 合成单核（conv 不落显存）。
// recurrent_state 由 state_bf16 决定精度：false=float*，true=__nv_bfloat16*（用 void* 传入）。
// mixed 为 in_proj 投影输出 [tokens, conv_dim]，布局 [q|k|v]。
void launch_linear_attention_fused(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                   const float *z, const float *b, const float *a, const float *a_log,
                                   const uint16_t *dt_bias, const float *norm_weight,
                                   void *recurrent_state, bool state_bf16, float *gated,
                                   int key_heads, int value_heads, int k_dim, int v_dim,
                                   int kernel, float eps, void *stream);

void launch_linear_attention_fused_batch(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                         const float *z, const float *b, const float *a, const float *a_log,
                                         const uint16_t *dt_bias, const float *norm_weight,
                                         void *recurrent_state, bool state_bf16, float *gated, int tokens,
                                         int key_heads, int value_heads, int k_dim, int v_dim,
                                         int kernel, float eps, void *stream);

// ================= 量化反量化 =================
// Q4_K 反量化：把 GGUF 的 Q4_K super-block（256 元素/144 字节）解到 f16 输出。
//   src           : device 端 Q4_K 原始字节（连续 BlockQ4K）；
//   out           : device 端 f16 输出（uint16_t 位模式），长度 >= num_elements；
//   num_elements  : 必须是 256 的整数倍。
// 埋点 dequant.q4k。用于“Q4_K 常驻 + gemm 前按需反量化到临时 f16 buffer”。
void launch_dequantize_q4k_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream);

// 其它 GGUF 量化类型反量化到 f16（均按 num_elements 计块）。
// Q8_0：block=32（34B），y=d*qs。Q5_0：block=32（22B），y=d*((q&0x1F)-16)。
// Q6_K：super-block=256（210B）。F32->f16 直转（norm/gate_inp 等 F32 权重上传用）。
void launch_dequantize_q80_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream);

void launch_dequantize_q50_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream);

void launch_dequantize_q6k_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream);

void launch_f32_to_f16_copy(const float *src, uint16_t *out, int64_t num_elements, void *stream);

void launch_f32_to_bf16_copy(const float *src, uint16_t *out, int64_t num_elements, void *stream);

// ================= MLA（多头潜在注意力，DeepSeek-V2-Lite）=================
// 维度约定：n_heads=16，qk_nope=128，qk_rope=64（解耦 RoPE 只旋转这 64 维），
//   qk_head=qk_nope+qk_rope=192，v_head=128，kv_lora=512。
// q 布局：[tokens, n_heads*qk_head]，每 head 前 qk_nope 为 nope、后 qk_rope 为 rope。
// kv_a 布局：[tokens, kv_lora + qk_rope]，前 kv_lora 为 latent、后 qk_rope 为共享 k_rope。
// inv_freq：device 端预计算的 RoPE 频率数组，长度 qk_rope/2（含 YARN 缩放），host 侧一次算好上传。
//
// (1) 对 kv_a 的 latent 段做 RMSNorm、对 k_rope 段做 RoPE，然后把 [latent||k_rope]
//     写入 latent KV cache（布局 [max_seq_len, kv_lora+qk_rope]）。
void launch_mla_kv_a(const float *kv_a, const float *kv_a_norm_weight,
                     float *output_kv_cache, int input_size, int kv_lora, int qk_rope,
                     int start_pos, const float *inv_freq, float eps, void *stream);

// (2) 对 q 的每个 token、每个 head 的 rope 段做 RoPE（nope 段不动）。q 原位更新。
void launch_mla_rope_q(float *q, int input_size, int n_heads, int qk_nope, int qk_rope,
                       int start_pos, const float *inv_freq, void *stream);

// (3) attend：kv_b_out 为已由 kv_b 投影解出的 per-(pos,head) k_nope||v，布局
//     [seq, n_heads*(qk_nope+v_head)]；k_rope 从 kv_cache 的 k_rope 段取（所有 head 共享）。
//     scores = (q_nope·k_nope + q_rope·k_rope)*softmax_scale，causal softmax，加权 v -> attn[tokens,n_heads*v_head]。
//     softmax_scale 已含 YARN 的 mscale^2（host 侧算好）。
void launch_mla_attend_batch(const float *q, const float *kv_b_out, const float *kv_cache,
                             float *attn, int input_size, int n_heads, int qk_nope, int qk_rope,
                             int v_head, int kv_lora, int start_pos,
                             float softmax_scale, void *stream);

// ================= MoE（DeepSeekMoE 路由）=================
// router_logits[tokens, n_experts] -> 每 token 选 top-k，对被选专家的 gate 值做 softmax
// 归一化并乘 routed_scaling_factor，输出 top_idx[tokens,k]（int）与 top_w[tokens,k]（float）。
void launch_moe_router_topk(const float *router_logits, int *top_idx, float *top_w,
                            int tokens, int n_experts, int k, float routed_scaling, void *stream);

// 把专家输出按权重累加到 hidden：out[token] += weight * expert_out[token]，n=hidden。
void launch_moe_accumulate(const float *expert_out, float weight, float *out, int n, void *stream);

// 同上，但权重从 device 读（*weight）：decode 时 top_w 留在 device，避免每层回读同步。
void launch_moe_accumulate_device(const float *expert_out, const float *weight, float *out, int n,
                                  void *stream);

// ================= argmax =================
// 对 logits[n] 求 argmax，把 token id 写到 device 端 out_idx[0]（greedy 用，结果留在 device）。
// 并列时取最小 index（与 CPU 端 Sampler::argmax 一致）。
void launch_argmax(const float *logits, int n, int *out_idx, void *stream);

#endif // LOCAL_LLM_KERNEL_CUH
