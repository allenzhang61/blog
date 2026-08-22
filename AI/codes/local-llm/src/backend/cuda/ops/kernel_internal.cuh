//
// Internal CUDA kernel declarations shared by kernel.cu and launch.cu.
// Public callers should include kernel.cuh instead.
//

#ifndef LOCAL_LLM_KERNEL_INTERNAL_CUH
#define LOCAL_LLM_KERNEL_INTERNAL_CUH

#include <cstdint>
#include <cuda_runtime.h>

constexpr int kBlock = 256;

inline cudaStream_t as_stream(void *stream) {
    return static_cast<cudaStream_t>(stream);
}

inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// ---- 逐元素 / Embedding / Norm ----
__global__ void add_kernel(const float *a, const float *b, float *out, int n);
__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int n);
__global__ void embedding_lookup_kernel(const int *input, float *output, const uint16_t *table,
                                        int vocab_size, int hidden_size, int weight_type);
__global__ void rms_norm_kernel(const float *input, float *output, const uint16_t *weight,
                                int weight_type, int hidden_size, float eps, bool one_plus);

// ---- full attention ----
__global__ void full_attention_q_kernel(const float *q_and_gate, const uint16_t *q_norm_weight,
                                        float *q, float *gate, int n_heads, int head_dim, int pos,
                                        float rope_theta, float partial_rotary_factor, float eps);
__global__ void full_attention_q_batch_kernel(const float *q_and_gate, const uint16_t *q_norm_weight,
                                              float *q, float *gate, int tokens, int n_heads,
                                              int head_dim, int start_pos, float rope_theta,
                                              float partial_rotary_factor, float eps);
__global__ void full_attention_kv_kernel(const float *k_in, const float *v_in,
                                         const uint16_t *k_norm_weight, float *key_cache,
                                         float *value_cache, int kv_heads, int head_dim,
                                         int max_seq_len, int pos, float rope_theta,
                                         float partial_rotary_factor, float eps);
__global__ void full_attention_kv_batch_kernel(const float *k_in, const float *v_in,
                                               const uint16_t *k_norm_weight, float *key_cache,
                                               float *value_cache, int tokens, int kv_heads,
                                               int head_dim, int max_seq_len, int start_pos,
                                               float rope_theta, float partial_rotary_factor, float eps);
__global__ void full_attention_attend_kernel(const float *q, const float *gate,
                                             const float *key_cache, const float *value_cache,
                                             float *attn, int n_heads, int kv_heads, int head_dim,
                                             int max_seq_len, int pos);
__global__ void full_attention_attend_batch_kernel(const float *q, const float *gate,
                                                   const float *key_cache, const float *value_cache,
                                                   float *attn, int tokens, int n_heads, int kv_heads,
                                                   int head_dim, int max_seq_len, int start_pos);

// ---- linear attention ----
__global__ void linear_attention_conv_kernel(const float *mixed, const uint16_t *conv_weight,
                                             float *conv_state, float *conv_out, int conv_dim,
                                             int kernel);
__global__ void linear_attention_conv_batch_kernel(const float *mixed, const uint16_t *conv_weight,
                                                   float *conv_state, float *conv_out, int tokens,
                                                   int conv_dim, int kernel);
__global__ void linear_attention_recurrent_kernel(const float *conv_out, const float *z,
                                                  const float *b, const float *a, const float *a_log,
                                                  const uint16_t *dt_bias, const float *norm_weight,
                                                  float *recurrent_state, float *gated,
                                                  int key_heads, int value_heads, int k_dim,
                                                  int v_dim, float eps);
__global__ void linear_attention_recurrent_batch_kernel(const float *conv_out, const float *z,
                                                        const float *b, const float *a,
                                                        const float *a_log, const uint16_t *dt_bias,
                                                        const float *norm_weight,
                                                        float *recurrent_state, float *gated,
                                                        int tokens, int key_heads, int value_heads,
                                                        int k_dim, int v_dim, float eps);

// ---- dtype / dequant ----
__global__ void dequantize_q4k_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void dequantize_q80_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void dequantize_q50_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void dequantize_q6k_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks);
__global__ void f32_to_f16_copy_kernel(const float *src, uint16_t *out, int64_t n);
__global__ void f32_to_bf16_copy_kernel(const float *src, uint16_t *out, int64_t n);

// ---- MLA ----
__global__ void mla_kv_a_kernel(const float *kv_a, const float *kv_a_norm_weight,
                                float *kv_cache, int kv_lora, int qk_rope, int pos,
                                const float *inv_freq, float eps);
__global__ void mla_kv_a_batch_kernel(const float *kv_a, const float *kv_a_norm_weight,
                                      float *kv_cache, int tokens, int kv_lora, int qk_rope,
                                      int start_pos, const float *inv_freq, float eps);
__global__ void mla_rope_q_kernel(float *q, int n_heads, int qk_nope, int qk_rope, int pos,
                                  const float *inv_freq);
__global__ void mla_rope_q_batch_kernel(float *q, int tokens, int n_heads, int qk_nope,
                                        int qk_rope, int start_pos, const float *inv_freq);
__global__ void mla_attend_kernel(const float *q, const float *kv_b_out, const float *kv_cache,
                                  float *attn, int n_heads, int qk_nope, int qk_rope,
                                  int v_head, int kv_lora, int pos, float softmax_scale);
__global__ void mla_attend_batch_kernel(const float *q, const float *kv_b_out,
                                        const float *kv_cache, float *attn, int tokens,
                                        int n_heads, int qk_nope, int qk_rope, int v_head,
                                        int kv_lora, int start_pos, float softmax_scale);

// ---- MoE ----
__global__ void moe_router_topk_kernel(const float *router_logits, int *top_idx, float *top_w,
                                       int tokens, int n_experts, int k, float routed_scaling);
__global__ void moe_accumulate_kernel(const float *expert_out, float weight, float *out, int n);

#endif // LOCAL_LLM_KERNEL_INTERNAL_CUH
