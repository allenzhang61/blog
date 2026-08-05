#pragma once

#include "../../core/config.h"
#include "../../core/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_inference {

// CUDA cuBLAS matvec，成功返回 true。
bool cuda_matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y);

// CUDA 路径计算 emb * hidden 后的 argmax token id。
bool cuda_argmax_matvec(const TensorRef & weight, const std::vector<float> & x, int & best_id);

// 获取复用的 CUDA hidden buffer，slot 用于 current/next 双缓冲。
void * cuda_token_hidden_buffer(int slot, int hidden_size);

// 获取存放生成 token ids 的 CUDA buffer。
void * cuda_generated_token_buffer(int count);

// 在设备端执行单 token embedding lookup，结果写入 device_out。
bool cuda_embedding_lookup_device(const TensorRef & emb, int token_id, void * device_out);

// 在设备端读取 token id 并执行 embedding lookup。
bool cuda_embedding_lookup_device_token(const TensorRef & emb, const void * device_token_id, void * device_out);

// 对设备端 hidden 做 final RMSNorm 和 logits argmax，结果回写主机 best_id。
bool cuda_final_norm_argmax_device(const TensorRef & norm_w, const TensorRef & emb, const void * device_hidden, int hidden_size, float eps, bool one_plus, int & best_id);

// 对设备端 hidden 做 final RMSNorm 和 logits argmax，结果保留在设备端 token buffer。
bool cuda_final_norm_argmax_to_device(const TensorRef & norm_w, const TensorRef & emb, const void * device_hidden, int hidden_size, float eps, bool one_plus, void * device_token_out);

// 将设备端生成 token ids 拷贝回主机 vector。
bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out);

// 同步当前 CUDA device，失败时返回 false。
bool cuda_synchronize_device();

// CUDA 批量 prefill 路径，返回最后一个 token 的设备端 hidden。
const void * cuda_prefill_batch(
    const ModelConfig & config,
    const ModelWeights & weights,
    const std::vector<int> & prompt_ids,
    std::vector<void *> & linear_states,
    std::vector<void *> & full_states,
    const std::vector<int> & full_max_seq_lens,
    int & seq_len);

// 当前构建/环境是否可用 CUDA cuBLAS matvec。
bool cuda_cublas_enabled();

// 当前构建/环境是否可用 fused MLP CUDA 路径。
bool cuda_fused_mlp_enabled();

// 当前构建/环境是否可用 attention projection CUDA 路径。
bool cuda_project_attention_enabled();

// 当前构建/环境是否可用整层 fused CUDA 路径。
bool cuda_full_layer_enabled();

// 当前构建/环境是否可用 RMSNorm + MLP fused CUDA 路径。
bool cuda_rmsnorm_mlp_enabled();

// 释放 linear attention 的 CUDA cache/state。
void cuda_free_linear_attention_state(void * state);

// 释放 full attention 的 CUDA KV cache/state。
void cuda_free_full_attention_state(void * state);

// CUDA fused MLP：gate/up/down 三个投影和 SiLU 乘法。
bool cuda_mlp_layer(const TensorRef & gate_w, const TensorRef & up_w, const TensorRef & down_w, const std::vector<float> & x, std::vector<float> & out);

// CUDA linear attention core，输入已完成各 projection。
bool cuda_linear_attention_layer(
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & norm_w,
    const TensorRef & out_w,
    const std::vector<float> & mixed,
    const std::vector<float> & z,
    const std::vector<float> & b,
    const std::vector<float> & a,
    void *& state,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    std::vector<float> & out);

// CUDA linear attention，包含输入 projection 和 attention core。
bool cuda_linear_attention_project_layer(
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    std::vector<float> & out);

// CUDA RMSNorm + linear attention projection fused 路径。
bool cuda_rmsnorm_linear_attention_project_layer(
    const TensorRef & input_norm_w,
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    bool one_plus,
    std::vector<float> & out);

// CUDA linear attention + post RMSNorm + MLP 的整层主机输入/输出路径。
bool cuda_linear_attention_full_layer(
    const TensorRef & input_norm_w,
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & attn_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const std::vector<float> & x,
    void *& state,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    bool one_plus,
    std::vector<float> & out);

// CUDA linear attention + post RMSNorm + MLP 的整层设备输入/输出路径。
bool cuda_linear_attention_full_layer_device(
    const TensorRef & input_norm_w,
    const TensorRef & in_proj_qkv_w,
    const TensorRef & in_proj_z_w,
    const TensorRef & in_proj_b_w,
    const TensorRef & in_proj_a_w,
    const TensorRef & conv_w,
    const TensorRef & a_log,
    const TensorRef & dt_bias,
    const TensorRef & attn_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const void * device_x,
    void * device_out,
    int hidden_dim,
    void *& state,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    int kernel,
    float eps,
    bool one_plus);

// CUDA full attention core，输入已完成 q/k/v projection。
bool cuda_full_attention_layer(
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & out_w,
    const std::vector<float> & q_and_gate,
    const std::vector<float> & k,
    const std::vector<float> & v,
    void *& state,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    std::vector<float> & out);

// CUDA full attention，包含 q/k/v projection 和 attention core。
bool cuda_full_attention_project_layer(
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    std::vector<float> & out);

// CUDA RMSNorm + full attention projection fused 路径。
bool cuda_rmsnorm_full_attention_project_layer(
    const TensorRef & input_norm_w,
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & out_w,
    const std::vector<float> & x,
    void *& state,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    bool one_plus,
    std::vector<float> & out);

// CUDA full attention + post RMSNorm + MLP 的整层主机输入/输出路径。
bool cuda_full_attention_full_layer(
    const TensorRef & input_norm_w,
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const std::vector<float> & x,
    void *& state,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    bool one_plus,
    std::vector<float> & out);

// CUDA full attention + post RMSNorm + MLP 的整层设备输入/输出路径。
bool cuda_full_attention_full_layer_device(
    const TensorRef & input_norm_w,
    const TensorRef & q_proj_w,
    const TensorRef & k_proj_w,
    const TensorRef & v_proj_w,
    const TensorRef & q_norm_w,
    const TensorRef & k_norm_w,
    const TensorRef & attn_out_w,
    const TensorRef & post_norm_w,
    const TensorRef & mlp_gate_w,
    const TensorRef & mlp_up_w,
    const TensorRef & mlp_down_w,
    const void * device_x,
    void * device_out,
    int hidden_dim,
    void *& state,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    bool one_plus);

// CUDA RMSNorm + MLP fused 路径。
bool cuda_rmsnorm_mlp_layer(
    const TensorRef & norm_w,
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
    const std::vector<float> & x,
    float eps,
    bool one_plus,
    std::vector<float> & out);

} // namespace llm_inference
