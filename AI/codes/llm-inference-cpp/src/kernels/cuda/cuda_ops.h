#pragma once

#include "../../core/config.h"
#include "../../model/weights.h"
#include "../../safetensors/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_inference {

// 获取复用的 CUDA hidden buffer，slot 用于 current/next 双缓冲。
void * cuda_token_hidden_buffer(int slot, int hidden_size);

// 获取复用的 CUDA token id buffer；decode 存 generated ids，prefill 存 prompt ids。
void * cuda_token_id_buffer(int count);

// 在设备端执行单 token embedding lookup，结果写入 device_out。
bool cuda_embedding_lookup_device(const TensorRef & emb, int token_id, void * device_out);

// 在设备端读取 token id 并执行 embedding lookup。
bool cuda_embedding_lookup_device_token(const TensorRef & emb, const void * device_token_id, void * device_out);

// 对设备端 hidden 做 final RMSNorm 和 logits argmax，结果保留在设备端 token buffer。
bool cuda_final_norm_argmax_to_device(const TensorRef & norm_w, const TensorRef & emb, const void * device_hidden, int hidden_size, float eps, bool one_plus, void * device_token_out);

// 将设备端生成 token ids 拷贝回主机 vector。
bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out);

// 同步当前 CUDA device，失败时返回 false。
bool cuda_synchronize_device();

// CUDA 批量 prefill 路径，在 device 上处理完整 prompt，返回最后一个 token 的 device hidden。
const void * cuda_prefill_batch(
    const ModelConfig & config,
    const ModelParams & params,
    const std::vector<int> & prompt_ids,
    std::vector<void *> & linear_states,
    std::vector<void *> & full_states,
    const std::vector<int> & full_max_seq_lens,
    int & seq_len);

// 释放 linear attention 的 CUDA cache/state。
void cuda_free_linear_attention_state(void * state);

// 释放 full attention 的 CUDA KV cache/state。
void cuda_free_full_attention_state(void * state);

// CUDA linear attention + post RMSNorm + MLP 的整层 device 输入/device 输出路径，避免中间拷回 host。
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

// CUDA full attention + post RMSNorm + MLP 的整层 device 输入/device 输出路径，decode 主路径使用。
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

} // namespace llm_inference
