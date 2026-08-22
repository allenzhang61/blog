//
// Created by zhangyoulun on 9/8/2026.
//

#include "kernel.cuh"
#include "kernel_internal.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "utils/stats/ScopedTimer.h"

// ================= launch 封装 =================

void launch_add(const float *a, const float *b, float *out, int n, void *stream) {
    ScopedGpuTimer timer("add", as_stream(stream));
    add_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(a, b, out, n);
}

void launch_silu_mul(const float *gate, const float *up, float *out, int n, void *stream) {
    ScopedGpuTimer timer("silu_mul", as_stream(stream));
    silu_mul_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(gate, up, out, n);
}

void launch_embedding_lookup(const int *input, float *output, const uint16_t *table,
                             int input_size, int vocab_size, int hidden_size, int weight_type, void *stream) {
    ScopedGpuTimer timer("embedding_lookup", as_stream(stream));
    embedding_lookup_kernel<<<input_size, kBlock, 0, as_stream(stream)>>>(
        input, output, table, vocab_size, hidden_size, weight_type);
}

/*
rows的值：
- layer 内 prefill norm：rows = input_size
- layer 内 decode norm：rows = 1
- final output norm：一直是 rows = 1
- Deepseek 的 decode 也是把单 token 包成 {1, hidden_size}，所以也是 1
后面如果支持真正 batch，比如 shape 变成 {batch, seq, hidden}，那当前 rows() 会返回 batch * seq，就不只 input_size / 1 了。
    但现在这套实现是单 batch，所以可以按 input_size 或 1 理解。
 *
 */
void launch_rms_norm(const float *input, float *output, const uint16_t *weight, int weight_type,
                     int rows, int hidden_size, float eps, bool one_plus, void *stream) {
    ScopedGpuTimer timer("rms_norm", as_stream(stream));
    rms_norm_kernel<<<rows, kBlock, kBlock * sizeof(float), as_stream(stream)>>>(
        input, output, weight, weight_type, hidden_size, eps, one_plus);
}

// ---- full attention ----

void launch_full_attention_q(const float *q_and_gate, const uint16_t *q_norm_weight,
                             float *q, float *gate, int n_heads, int head_dim, int pos,
                             float rope_theta, float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_q", as_stream(stream));
    full_attention_q_kernel<<<n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, n_heads, head_dim, pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_q_batch(const float *q_and_gate, const uint16_t *q_norm_weight,
                                   float *q, float *gate, int tokens, int n_heads, int head_dim,
                                   int start_pos, float rope_theta, float partial_rotary_factor,
                                   float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_q_batch", as_stream(stream));
    full_attention_q_batch_kernel<<<tokens * n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, tokens, n_heads, head_dim, start_pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_kv(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                              float *key_cache, float *value_cache, int kv_heads, int head_dim,
                              int max_seq_len, int pos, float rope_theta, float partial_rotary_factor,
                              float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_kv", as_stream(stream));
    full_attention_kv_kernel<<<kv_heads, kBlock, 0, as_stream(stream)>>>(
        k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim, max_seq_len, pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_kv_batch(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                                    float *key_cache, float *value_cache, int tokens, int kv_heads,
                                    int head_dim, int max_seq_len, int start_pos, float rope_theta,
                                    float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_kv_batch", as_stream(stream));
    full_attention_kv_batch_kernel<<<tokens * kv_heads, kBlock, 0, as_stream(stream)>>>(
        k_in, v_in, k_norm_weight, key_cache, value_cache, tokens, kv_heads, head_dim, max_seq_len,
        start_pos, rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_attend(const float *q, const float *gate, const float *key_cache,
                                  const float *value_cache, float *attn, int n_heads, int kv_heads,
                                  int head_dim, int max_seq_len, int pos, void *stream) {
    ScopedGpuTimer timer("full_attention_attend", as_stream(stream));
    size_t smem = (static_cast<size_t>(pos + 1) + kBlock) * sizeof(float);
    full_attention_attend_kernel<<<n_heads, kBlock, smem, as_stream(stream)>>>(
        q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim, max_seq_len, pos);
}

void launch_full_attention_attend_batch(const float *q, const float *gate, const float *key_cache,
                                        const float *value_cache, float *attn, int tokens, int n_heads,
                                        int kv_heads, int head_dim, int max_seq_len, int start_pos,
                                        void *stream) {
    ScopedGpuTimer timer("full_attention_attend_batch", as_stream(stream));
    // shared 大小需覆盖本段最大位置的 scores。
    size_t max_pos = static_cast<size_t>(start_pos + tokens - 1);
    size_t smem = (max_pos + 1 + kBlock) * sizeof(float);
    full_attention_attend_batch_kernel<<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
        q, gate, key_cache, value_cache, attn, tokens, n_heads, kv_heads, head_dim, max_seq_len, start_pos);
}

// ---- linear attention ----

void launch_linear_attention_conv(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                  float *conv_out, int conv_dim, int kernel, void *stream) {
    ScopedGpuTimer timer("linear_attention_conv", as_stream(stream));
    linear_attention_conv_kernel<<<grid_for(conv_dim), kBlock, 0, as_stream(stream)>>>(
        mixed, conv_weight, conv_state, conv_out, conv_dim, kernel);
}

void launch_linear_attention_conv_batch(const float *mixed, const uint16_t *conv_weight,
                                        float *conv_state, float *conv_out, int tokens, int conv_dim,
                                        int kernel, void *stream) {
    ScopedGpuTimer timer("linear_attention_conv_batch", as_stream(stream));
    linear_attention_conv_batch_kernel<<<grid_for(conv_dim), kBlock, 0, as_stream(stream)>>>(
        mixed, conv_weight, conv_state, conv_out, tokens, conv_dim, kernel);
}

void launch_linear_attention_recurrent(const float *conv_out, const float *z, const float *b,
                                       const float *a, const float *a_log, const uint16_t *dt_bias,
                                       const float *norm_weight, float *recurrent_state, float *gated,
                                       int key_heads, int value_heads, int k_dim, int v_dim,
                                       float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_recurrent", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kBlock) * sizeof(float);
    linear_attention_recurrent_kernel<<<value_heads, kBlock, smem, as_stream(stream)>>>(
        conv_out, z, b, a, a_log, dt_bias, norm_weight, recurrent_state, gated,
        key_heads, value_heads, k_dim, v_dim, eps);
}

void launch_linear_attention_recurrent_batch(const float *conv_out, const float *z, const float *b,
                                             const float *a, const float *a_log, const uint16_t *dt_bias,
                                             const float *norm_weight, float *recurrent_state, float *gated,
                                             int tokens, int key_heads, int value_heads, int k_dim, int v_dim,
                                             float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_recurrent_batch", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kBlock) * sizeof(float);
    linear_attention_recurrent_batch_kernel<<<value_heads, kBlock, smem, as_stream(stream)>>>(
        conv_out, z, b, a, a_log, dt_bias, norm_weight, recurrent_state, gated, tokens,
        key_heads, value_heads, k_dim, v_dim, eps);
}

void launch_dequantize_q4k_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q4k_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 256; // 调用方保证 num_elements % 256 == 0
    dequantize_q4k_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 128, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_dequantize_q80_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q80_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 32;
    dequantize_q80_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 32, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_dequantize_q50_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q50_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 32;
    dequantize_q50_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 16, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_dequantize_q6k_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q6k_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 256;
    dequantize_q6k_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 32, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_f32_to_f16_copy(const float *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("f32_to_f16_copy", as_stream(stream));
    f32_to_f16_copy_kernel<<<grid_for(static_cast<int>(num_elements)), kBlock, 0, as_stream(stream)>>>(
        src, out, num_elements);
}

void launch_f32_to_bf16_copy(const float *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("f32_to_bf16_copy", as_stream(stream));
    f32_to_bf16_copy_kernel<<<grid_for(static_cast<int>(num_elements)), kBlock, 0, as_stream(stream)>>>(
        src, out, num_elements);
}

// ---- MLA ----

void launch_mla_kv_a(const float *kv_a, const float *kv_a_norm_weight, float *kv_cache,
                     int kv_lora, int qk_rope, int max_seq_len, int pos,
                     const float *inv_freq, float eps, void *stream) {
    ScopedGpuTimer timer("mla_kv_a", as_stream(stream));
    (void) max_seq_len;
    size_t smem = (static_cast<size_t>(kBlock) + kv_lora + qk_rope) * sizeof(float);
    mla_kv_a_kernel<<<1, kBlock, smem, as_stream(stream)>>>(
        kv_a, kv_a_norm_weight, kv_cache, kv_lora, qk_rope, pos, inv_freq, eps);
}

void launch_mla_kv_a_batch(const float *kv_a, const float *kv_a_norm_weight, float *kv_cache,
                           int tokens, int kv_lora, int qk_rope, int max_seq_len, int start_pos,
                           const float *inv_freq, float eps, void *stream) {
    ScopedGpuTimer timer("mla_kv_a_batch", as_stream(stream));
    (void) max_seq_len;
    size_t smem = (static_cast<size_t>(kBlock) + kv_lora + qk_rope) * sizeof(float);
    mla_kv_a_batch_kernel<<<tokens, kBlock, smem, as_stream(stream)>>>(
        kv_a, kv_a_norm_weight, kv_cache, tokens, kv_lora, qk_rope, start_pos, inv_freq, eps);
}

void launch_mla_rope_q(float *q, int n_heads, int qk_nope, int qk_rope, int pos,
                       const float *inv_freq, void *stream) {
    ScopedGpuTimer timer("mla_rope_q", as_stream(stream));
    mla_rope_q_kernel<<<n_heads, kBlock, 0, as_stream(stream)>>>(
        q, n_heads, qk_nope, qk_rope, pos, inv_freq);
}

void launch_mla_rope_q_batch(float *q, int tokens, int n_heads, int qk_nope, int qk_rope,
                             int start_pos, const float *inv_freq, void *stream) {
    ScopedGpuTimer timer("mla_rope_q_batch", as_stream(stream));
    mla_rope_q_batch_kernel<<<tokens * n_heads, kBlock, 0, as_stream(stream)>>>(
        q, tokens, n_heads, qk_nope, qk_rope, start_pos, inv_freq);
}

void launch_mla_attend(const float *q, const float *kv_b_out, const float *kv_cache, float *attn,
                       int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                       int max_seq_len, int pos, float softmax_scale, void *stream) {
    ScopedGpuTimer timer("mla_attend", as_stream(stream));
    (void) max_seq_len;
    size_t smem = (static_cast<size_t>(pos + 1) + kBlock) * sizeof(float);
    mla_attend_kernel<<<n_heads, kBlock, smem, as_stream(stream)>>>(
        q, kv_b_out, kv_cache, attn, n_heads, qk_nope, qk_rope, v_head, kv_lora, pos, softmax_scale);
}

void launch_mla_attend_batch(const float *q, const float *kv_b_out, const float *kv_cache, float *attn,
                             int tokens, int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                             int max_seq_len, int start_pos, float softmax_scale, void *stream) {
    ScopedGpuTimer timer("mla_attend_batch", as_stream(stream));
    (void) max_seq_len;
    size_t max_pos = static_cast<size_t>(start_pos + tokens - 1);
    size_t smem = (max_pos + 1 + kBlock) * sizeof(float);
    mla_attend_batch_kernel<<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
        q, kv_b_out, kv_cache, attn, tokens, n_heads, qk_nope, qk_rope, v_head, kv_lora,
        start_pos, softmax_scale);
}

// ---- MoE ----

void launch_moe_router_topk(const float *router_logits, int *top_idx, float *top_w,
                            int tokens, int n_experts, int k, float routed_scaling, void *stream) {
    ScopedGpuTimer timer("moe_router_topk", as_stream(stream));
    moe_router_topk_kernel<<<tokens, 32, 0, as_stream(stream)>>>(
        router_logits, top_idx, top_w, tokens, n_experts, k, routed_scaling);
}

void launch_moe_accumulate(const float *expert_out, float weight, float *out, int n, void *stream) {
    ScopedGpuTimer timer("moe_accumulate", as_stream(stream));
    moe_accumulate_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(expert_out, weight, out, n);
}
