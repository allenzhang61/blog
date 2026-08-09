#pragma once

#include <cstddef>
#include <cstdint>

namespace llm_inference {

// 启动 SiLU(gate) * up 的逐元素 kernel。
void launch_silu_mul(const float * gate, const float * up, float * out, int n, void * stream);

// 启动批量 gate/up 拼接输入的 SiLU 乘法 kernel。
void launch_silu_mul_gate_up_batch(const float * gate_up, float * out, int tokens, int intermediate, void * stream);

// 将 float 数组转换为 BF16。
void launch_float_to_bf16(const float * input, uint16_t * output, int n, void * stream);

// 将 float 数组转换为 F16。
void launch_float_to_f16(const float * input, uint16_t * output, int n, void * stream);

// 将一行低精度 BF16/F16 权重转换为 float。
void launch_lowp_row_to_float(const uint16_t * input, float * output, int n, int lowp_type, void * stream);

// 按设备端 token id 读取 embedding 行并转换为 float。
void launch_lowp_embedding_id_to_float(const uint16_t * input, const int * token_id, float * output, int vocab, int hidden, int lowp_type, void * stream);

// 拷贝一个 int 值。
void launch_copy_int(const int * input, int * output, void * stream);

// 批量读取 embedding 行并转换为 float。
void launch_embedding_batch_to_float(const uint16_t * emb, const int * token_ids, float * output, int tokens, int vocab, int hidden, int lowp_type, void * stream);

// 批量 RMSNorm，并将结果写为 BF16。
void launch_rms_norm_batch_to_bf16(const float * input, const uint16_t * weight, uint16_t * output, int tokens, int hidden, float eps, bool one_plus, void * stream);

// 单向量 RMSNorm，并将结果写为 F16。
void launch_rms_norm_to_f16(const float * input, const uint16_t * weight, uint16_t * output, int n, float eps, bool one_plus, void * stream);

// 单向量 RMSNorm，并将结果写为 BF16。
void launch_rms_norm_to_bf16(const float * input, const uint16_t * weight, uint16_t * output, int n, float eps, bool one_plus, void * stream);

// linear attention 批量 depthwise conv。
void launch_linear_attention_conv_batch(
    const float * mixed,
    const uint16_t * conv_weight,
    float * conv_state,
    float * conv_out,
    int tokens,
    int conv_dim,
    int kernel,
    void * stream);

// linear attention 批量 recurrent state 更新和 gated 输出。
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

// linear attention 单 token depthwise conv。
void launch_linear_attention_conv(
    const float * mixed,
    const uint16_t * conv_weight,
    float * conv_state,
    float * conv_out,
    int conv_dim,
    int kernel,
    void * stream);

// linear attention 单 token recurrent state 更新和 gated 输出。
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

// full attention 单 token query/gate 归一化和 RoPE。
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

// full attention 单 token key/value 归一化、RoPE 和 KV cache 写入。
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

// full attention 单 token causal attention 计算。
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

// full attention 批量 query/gate 归一化和 RoPE。
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

// full attention 批量 key/value 归一化、RoPE 和 KV cache 写入。
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

// full attention 批量 causal attention 计算。
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

// 对 float 数组做分块 argmax。
void launch_argmax_float(
    const float * values,
    int n,
    float * block_values,
    int * block_indices,
    float * best_value,
    int * best_index,
    int blocks,
    void * stream);

// BF16 权重和 BF16 输入的矩阵向量乘。
void launch_bf16_matvec(
    const uint16_t * weight,
    const uint16_t * x,
    float * y,
    int rows,
    int cols,
    void * stream);

// 单向量逐元素相加。
void launch_add_float(const float * a, const float * b, float * out, int n, void * stream);

// 批量逐元素相加。
void launch_add_float_batch(const float * a, const float * b, float * out, int n, void * stream);

// 批量 residual add + RMSNorm，并将 norm 结果写为 BF16。
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

// 单向量 residual add + RMSNorm，并将 norm 结果写为 BF16。
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
