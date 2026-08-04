#pragma once

#include <cstddef>
#include <cstdint>

namespace llm_inference {

void launch_silu_mul(const float * gate, const float * up, float * out, int n, void * stream);
void launch_silu_mul_gate_up_batch(const float * gate_up, float * out, int tokens, int intermediate, void * stream);
void launch_float_to_bf16(const float * input, uint16_t * output, int n, void * stream);
void launch_float_to_f16(const float * input, uint16_t * output, int n, void * stream);
void launch_lowp_row_to_float(const uint16_t * input, float * output, int n, int lowp_type, void * stream);
void launch_lowp_embedding_id_to_float(const uint16_t * input, const int * token_id, float * output, int vocab, int hidden, int lowp_type, void * stream);
void launch_copy_int(const int * input, int * output, void * stream);
void launch_embedding_batch_to_float(const uint16_t * emb, const int * token_ids, float * output, int tokens, int vocab, int hidden, int lowp_type, void * stream);
void launch_rms_norm_batch_to_bf16(const float * input, const uint16_t * weight, uint16_t * output, int tokens, int hidden, float eps, bool one_plus, void * stream);
void launch_rms_norm_to_f16(const float * input, const uint16_t * weight, uint16_t * output, int n, float eps, bool one_plus, void * stream);
void launch_rms_norm_to_bf16(const float * input, const uint16_t * weight, uint16_t * output, int n, float eps, bool one_plus, void * stream);
void launch_linear_attention_conv_batch(
    const float * mixed,
    const uint16_t * conv_weight,
    float * conv_state,
    float * conv_out,
    int tokens,
    int conv_dim,
    int kernel,
    void * stream);
void launch_linear_attention_recurrent_batch(
    const float * conv_out,
    const float * z,
    const float * b,
    const float * a,
    const float * a_log,
    const uint16_t * dt_bias,
    const float * norm_weight,
    float * recurrent_state,
    float * gated,
    int tokens,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    float eps,
    void * stream);
void launch_linear_attention_conv(
    const float * mixed,
    const uint16_t * conv_weight,
    float * conv_state,
    float * conv_out,
    int conv_dim,
    int kernel,
    void * stream);
void launch_linear_attention_recurrent(
    const float * conv_out,
    const float * z,
    const float * b,
    const float * a,
    const float * a_log,
    const uint16_t * dt_bias,
    const float * norm_weight,
    float * recurrent_state,
    float * gated,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    float eps,
    void * stream);
void launch_full_attention_q(
    const float * q_and_gate,
    const uint16_t * q_norm_weight,
    float * q,
    float * gate,
    int n_heads,
    int head_dim,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    void * stream);
void launch_full_attention_kv(
    const float * k_in,
    const float * v_in,
    const uint16_t * k_norm_weight,
    float * key_cache,
    float * value_cache,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    void * stream);
void launch_full_attention_attend(
    const float * q,
    const float * gate,
    const float * key_cache,
    const float * value_cache,
    float * attn,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    void * stream);
void launch_full_attention_q_batch(
    const float * q_and_gate,
    const uint16_t * q_norm_weight,
    float * q,
    float * gate,
    int tokens,
    int n_heads,
    int head_dim,
    int start_pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    void * stream);
void launch_full_attention_kv_batch(
    const float * k_in,
    const float * v_in,
    const uint16_t * k_norm_weight,
    float * key_cache,
    float * value_cache,
    int tokens,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    void * stream);
void launch_full_attention_attend_batch(
    const float * q,
    const float * gate,
    const float * key_cache,
    const float * value_cache,
    float * attn,
    int tokens,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    void * stream);
void launch_argmax_float(
    const float * values,
    int n,
    float * block_values,
    int * block_indices,
    float * best_value,
    int * best_index,
    int blocks,
    void * stream);
void launch_bf16_matvec(
    const uint16_t * weight,
    const uint16_t * x,
    float * y,
    int rows,
    int cols,
    void * stream);
void launch_add_float(const float * a, const float * b, float * out, int n, void * stream);
void launch_add_float_batch(const float * a, const float * b, float * out, int n, void * stream);
void launch_add_rms_norm_batch_to_bf16(
    const float * a,
    const float * b,
    const uint16_t * weight,
    float * sum_out,
    uint16_t * norm_out,
    int tokens,
    int hidden,
    float eps,
    bool one_plus,
    void * stream);
void launch_add_rms_norm_to_bf16(
    const float * a,
    const float * b,
    const uint16_t * weight,
    float * sum_out,
    uint16_t * norm_out,
    int n,
    float eps,
    bool one_plus,
    void * stream);

} // namespace llm_inference
