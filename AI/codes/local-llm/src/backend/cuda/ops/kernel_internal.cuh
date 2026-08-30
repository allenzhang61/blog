//
// Internal CUDA kernel declarations shared by kernel.cu and launch.cu.
// Public callers should include kernel.cuh instead.
//

#ifndef LOCAL_LLM_KERNEL_INTERNAL_CUH
#define LOCAL_LLM_KERNEL_INTERNAL_CUH

#include <cstdint>
#include <cuda_runtime.h>

constexpr int kBlock = 256;
// linear attention 递归 kernel 每个 head 一个 block，state 有 k_dim*v_dim=16384 个元素，
// 用更大的 block（512 线程）减少每线程串行迭代次数，提高 SM 内延迟隐藏。
constexpr int kLinearRecurBlock = 512;

inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// ---- 逐元素 / Embedding / Norm ----
__global__ void bf16_gemv_kernel(const uint16_t *weight, const uint16_t *x, float *y,
                                 int out_dim, int in_dim);
// 量化直算 GEMM（decode M=1 / prefill M>1）：QUANT_TYPE 12=Q4_K 14=Q6_K 6=Q5_0 8=Q8_0。
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void quant_gemv_kernel(const uint8_t *weight, size_t row_bytes, const float *x,
                                  float *y, int out_dim, int in_dim, int m);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void quant_gemv_add_kernel(const uint8_t *weight, size_t row_bytes, const float *x,
                                      float *y, int out_dim, int in_dim);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void quant_matmul_kernel(const uint8_t *weight, size_t row_bytes, const float *x,
                                    float *y, int out_dim, int in_dim, int m);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void quant_swiglu_kernel(const uint8_t *gate_weight, const uint8_t *up_weight,
                                    size_t gate_row_bytes, size_t up_row_bytes,
                                    const float *x, float *act, int ffn_dim, int in_dim, bool fast_silu);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void quant_swiglu_indexed_kernel(const uint8_t *gate_weight, const uint8_t *up_weight,
                                            size_t gate_expert_bytes, size_t up_expert_bytes,
                                            size_t gate_row_bytes, size_t up_row_bytes, const float *x,
                                            const int *expert_ids, float *act, int k, int ffn_dim, int in_dim,
                                            bool fast_silu);
template <int QUANT_TYPE>
__global__ void quant_swiglu_indexed_block_kernel(const uint8_t *gate_weight, const uint8_t *up_weight,
                                                  size_t gate_expert_bytes, size_t up_expert_bytes,
                                                  size_t gate_row_bytes, size_t up_row_bytes, const float *x,
                                                  const int *expert_ids, float *act, int k, int ffn_dim, int in_dim,
                                                  int blocks_per_row, bool fast_silu);
template <int QUANT_TYPE>
__global__ void quant_swiglu_indexed_q8_1_kernel(const uint8_t *gate_weight, const uint8_t *up_weight,
                                                 size_t gate_expert_bytes, size_t up_expert_bytes,
                                                 size_t gate_row_bytes, size_t up_row_bytes, const uint8_t *x_q8_1,
                                                 const int *expert_ids, float *act, int k, int ffn_dim, int in_dim,
                                                 int blocks_per_row);
__global__ void quantize_q8_1_kernel(const float *x, uint8_t *x_q8_1, int in_dim, int m, int blocks_per_row,
                                     bool store_raw_sum);
__global__ void quantize_q8_1_mmq_kernel(const float *x, uint8_t *x_q8_1, int in_dim, int m, int groups_per_row,
                                         bool store_raw_sum);
template <int QUANT_TYPE>
__global__ void quant_gemv_q8_1_kernel(const uint8_t *weight, size_t row_bytes, const uint8_t *x_q8_1,
                                       float *y, int out_dim, int in_dim, int blocks_per_row);
template <int QUANT_TYPE>
__global__ void quant_matmul_q8_1_kernel(const uint8_t *weight, size_t row_bytes, const uint8_t *x_q8_1,
                                         float *y, int out_dim, int in_dim, int m, int blocks_per_row);
template <int QUANT_TYPE>
__global__ void quant_matmul_q8_1_mmq_kernel(const uint8_t *weight, size_t row_bytes, const uint8_t *x_q8_1,
                                             float *y, int out_dim, int in_dim, int m, int groups_per_row);
template <int QUANT_TYPE>
__global__ void quant_down_q8_1_indexed_accum_kernel(const uint8_t *down_weight, size_t down_expert_bytes,
                                                     size_t down_row_bytes, const uint8_t *act_q8_1,
                                                     const int *expert_ids, const float *route_weights,
                                                     float *out, int k, int hidden_size, int ffn_dim,
                                                     int blocks_per_row);
template <int QUANT_TYPE>
__global__ void quant_down_q8_1_indexed_accum_shared_kernel(const uint8_t *down_weight, size_t down_expert_bytes,
                                                            size_t down_row_bytes, const uint8_t *act_q8_1,
                                                            const int *expert_ids, const float *route_weights,
                                                            float *out, int k, int hidden_size, int ffn_dim,
                                                            int blocks_per_row);
template <int QUANT_TYPE>
__global__ void quant_down_f32_indexed_accum_kernel(const uint8_t *down_weight, size_t down_expert_bytes,
                                                    size_t down_row_bytes, const float *act,
                                                    const int *expert_ids, const float *route_weights,
                                                    float *out, int k, int hidden_size, int ffn_dim,
                                                    int blocks_per_row);
// 量化直算 Embedding：按 token id 只反量化命中行到 f32，避免整表展开成 F16。
template <int QUANT_TYPE>
__global__ void quant_embedding_kernel(const int *input, float *output, const uint8_t *table,
                                       size_t row_bytes, int hidden_size, int input_size);
__global__ void add_kernel(const float *a, const float *b, float *out, int n);
__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int n);
__global__ void embedding_lookup_kernel(const int *input, float *output, const uint16_t *table,
                                        int vocab_size, int hidden_size, int weight_type);
__global__ void rms_norm_kernel(const float *input, float *output, const uint16_t *weight,
                                int weight_type, int hidden_size, float eps, bool one_plus);
__global__ void add_rms_norm_kernel(const float *x, const float *residual, float *out_residual,
                                    float *out_norm, const uint16_t *weight, int weight_type,
                                    int hidden_size, float eps, bool one_plus);

// ---- full attention ----
__global__ void full_attention_q_kernel(const float *q_and_gate, const uint16_t *q_norm_weight,
                                        float *q, float *gate, int n_heads, int head_dim,
                                        const int *pos_dev,
                                        float rope_theta, float partial_rotary_factor, float eps);
__global__ void full_attention_q_batch_kernel(const float *q_and_gate, const uint16_t *q_norm_weight,
                                              float *q, float *gate, int tokens, int n_heads,
                                              int head_dim, int start_pos, float rope_theta,
                                              float partial_rotary_factor, float eps);
// full attention KV / attend kernel（模板：KvT=float 或 __nv_bfloat16，KV cache 存储精度）。
template <typename KvT>
__global__ void full_attention_kv_kernel(const float *k_in, const float *v_in,
                                         const uint16_t *k_norm_weight, KvT *key_cache,
                                         KvT *value_cache, int kv_heads, int head_dim,
                                         int max_seq_len, const int *pos_dev, float rope_theta,
                                         float partial_rotary_factor, float eps);
template <typename KvT>
__global__ void full_attention_kv_batch_kernel(const float *k_in, const float *v_in,
                                               const uint16_t *k_norm_weight, KvT *key_cache,
                                               KvT *value_cache, int tokens, int kv_heads,
                                               int head_dim, int max_seq_len, int start_pos,
                                               float rope_theta, float partial_rotary_factor, float eps);
template <typename KvT>
__global__ void full_attention_attend_kernel(const float *q, const float *gate,
                                             const KvT *key_cache, const KvT *value_cache,
                                             float *attn, int n_heads, int kv_heads, int head_dim,
                                             int max_seq_len, const int *pos_dev);
template <typename KvT>
__global__ void full_attention_attend_batch_kernel(const float *q, const float *gate,
                                                   const KvT *key_cache, const KvT *value_cache,
                                                   float *attn, int tokens, int n_heads, int kv_heads,
                                                   int head_dim, int max_seq_len, int start_pos);

// ---- linear attention ----
__global__ void linear_attention_conv_kernel(const float *mixed, const uint16_t *conv_weight,
                                             float *conv_state, float *conv_out, int conv_dim,
                                             int kernel);
__global__ void linear_attention_conv_batch_kernel(const float *mixed, const uint16_t *conv_weight,
                                                   float *conv_state, float *conv_out, int tokens,
                                                   int conv_dim, int kernel);
// 递归 kernel（模板：StateT=float 或 __nv_bfloat16）。定义与显式实例化在 kernel.cu。
template <typename StateT>
__global__ void linear_attention_recurrent_kernel(const float *conv_out, const float *z,
                                                  const float *b, const float *a, const float *a_log,
                                                  const uint16_t *dt_bias, const float *norm_weight,
                                                  StateT *recurrent_state, float *gated,
                                                  int key_heads, int value_heads, int k_dim,
                                                  int v_dim, float eps);
__global__ void linear_attention_recurrent_batch_kernel(const float *conv_out, const float *z,
                                                        const float *b, const float *a,
                                                        const float *a_log, const uint16_t *dt_bias,
                                                        const float *norm_weight,
                                                        float *recurrent_state, float *gated,
                                                        int tokens, int key_heads, int value_heads,
                                                        int k_dim, int v_dim, float eps);
// 融合 kernel（模板：StateT=float 或 __nv_bfloat16）。声明为模板，定义与显式实例化在 kernel.cu。
template <typename StateT>
__global__ void linear_attention_fused_kernel(const float *mixed, const uint16_t *conv_weight,
                                              float *conv_state, const float *z, const float *b,
                                              const float *a, const float *a_log,
                                              const uint16_t *dt_bias, const float *norm_weight,
                                              StateT *recurrent_state, float *gated,
                                              int key_heads, int value_heads, int k_dim, int v_dim,
                                              int kernel, float eps);
template <typename StateT>
__global__ void linear_attention_fused_batch_kernel(const float *mixed, const uint16_t *conv_weight,
                                                    float *conv_state, const float *z, const float *b,
                                                    const float *a, const float *a_log,
                                                    const uint16_t *dt_bias, const float *norm_weight,
                                                    StateT *recurrent_state, float *gated, int tokens,
                                                    int key_heads, int value_heads, int k_dim, int v_dim,
                                                    int kernel, float eps);

// ---- dtype / dequant ----
__global__ void dequantize_q4k_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void dequantize_q80_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void dequantize_q50_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void dequantize_q6k_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void f32_to_f16_copy_kernel(const float *src, uint16_t *out, int64_t n);
__global__ void f32_to_bf16_copy_kernel(const float *src, uint16_t *out, int64_t n);

// ---- MLA ----
__global__ void mla_kv_a_kernel(const float *kv_a, const float *kv_a_norm_weight,
                                float *output_kv_cache, int input_size, int kv_lora, int qk_rope,
                                int start_pos, const float *inv_freq, float eps);
__global__ void mla_kv_a_device_pos_kernel(const float *kv_a, const float *kv_a_norm_weight,
                                           float *output_kv_cache, int input_size, int kv_lora, int qk_rope,
                                           const int *d_pos, const float *inv_freq, float eps);
__global__ void mla_rope_q_kernel(float *q, int input_size, int n_heads, int qk_nope,
                                  int qk_rope, int start_pos, const float *inv_freq);
__global__ void mla_rope_q_device_pos_kernel(float *q, int input_size, int n_heads, int qk_nope,
                                             int qk_rope, const int *d_pos, const float *inv_freq);
__global__ void mla_attend_batch_kernel(const float *q, const float *kv_b_out,
                                        const float *kv_cache, float *attn, int tokens,
                                        int n_heads, int qk_nope, int qk_rope, int v_head,
                                        int kv_lora, int start_pos, float softmax_scale);
__global__ void mla_attend_batch_device_pos_kernel(const float *q, const float *kv_b_out,
                                                   const float *kv_cache, float *attn, int tokens,
                                                   int n_heads, int qk_nope, int qk_rope, int v_head,
                                                   int kv_lora, const int *d_pos, float softmax_scale);
__global__ void mla_gather_latent_device_pos_kernel(const float *kv_cache, float *latent,
                                                    int kv_lora, int qk_rope, const int *d_pos);
__global__ void mla_store_kv_b_device_pos_kernel(const float *kv_b_new, float *kv_b_cache,
                                                 int kvb_out, const int *d_pos);
__global__ void mla_store_latent_q8_1_device_pos_kernel(const float *kv_cache, uint8_t *latent_q8_1_cache,
                                                        int kv_lora, int qk_rope, size_t row_bytes,
                                                        int blocks_per_row, const int *d_pos);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void mla_absorb_q_nope_kernel(const float *q, const uint8_t *kv_b_weight,
                                         size_t row_bytes, float *q_abs, int n_heads,
                                         int qk_nope, int qk_rope, int v_head, int kv_lora);
template <bool F16_OPERANDS>
__global__ void mla_absorb_q4_xsum_delta_kernel(const float *q, const uint8_t *kv_b_weight,
                                                size_t row_bytes, float *q_abs_xsum_delta,
                                                int n_heads, int qk_nope, int qk_rope,
                                                int v_head, int blocks_per_row);
__global__ void mla_absorb_attend_device_pos_kernel(const float *q_abs, const float *q,
                                                    const uint8_t *latent_q8_1_cache,
                                                    size_t latent_q8_1_row_bytes,
                                                    const float *q_abs_xsum_delta, float *attn_xsum_delta,
                                                    const float *kv_cache,
                                                    float *attn_latent,
                                                    int n_heads, int qk_nope, int qk_rope, int kv_lora,
                                                    const int *d_pos, float softmax_scale);
__global__ void mla_absorb_scores_device_pos_kernel(const float *q_abs, const float *q,
                                                    const uint8_t *latent_q8_1_cache,
                                                    size_t latent_q8_1_row_bytes,
                                                    const float *q_abs_xsum_delta, const float *kv_cache,
                                                    float *scores, int n_heads, int qk_nope, int qk_rope,
                                                    int kv_lora, const int *d_pos, int max_seq_len,
                                                    float softmax_scale);
__global__ void mla_absorb_context_device_pos_kernel(const float *scores, const uint8_t *latent_q8_1_cache,
                                                     size_t latent_q8_1_row_bytes, float *attn_xsum_delta,
                                                     float *attn_latent, int n_heads, int kv_lora,
                                                     const int *d_pos, int max_seq_len);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void mla_project_v_device_pos_kernel(const uint8_t *kv_b_weight, size_t row_bytes,
                                                const uint8_t *latent_q8_1_cache, size_t latent_q8_1_row_bytes,
                                                float *kv_b_cache, int n_heads, int qk_nope, int v_head,
                                                int kv_lora, const int *d_pos);
__global__ void mla_absorb_context_v_device_pos_kernel(const float *scores, const float *kv_b_cache,
                                                       float *attn, int n_heads, int qk_nope, int v_head,
                                                       const int *d_pos, int max_seq_len);
template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void mla_absorb_v_kernel(const uint8_t *kv_b_weight, size_t row_bytes,
                                    const float *attn_latent, const float *attn_xsum_delta, float *attn,
                                    int n_heads, int qk_nope, int v_head, int kv_lora);

// ---- MoE ----
__global__ void moe_router_topk_kernel(const float *router_logits, int *top_idx, float *top_w,
                                       int tokens, int n_experts, int k, float routed_scaling);
__global__ void moe_accumulate_kernel(const float *expert_out, float weight, float *out, int n);
__global__ void moe_accumulate_device_kernel(const float *expert_out, const float *weight, float *out, int n);

// ---- argmax ----
// 两阶段归约：阶段1 多 block 各求局部 argmax，阶段2 单 block 归约局部结果。
__global__ void argmax_partial_kernel(const float *logits, int n, float *partial_vals,
                                      int *partial_idxs);
__global__ void argmax_final_kernel(const float *partial_vals, const int *partial_idxs,
                                    int num_partials, int *out_idx);

#endif // LOCAL_LLM_KERNEL_INTERNAL_CUH
