//
// Created by zhangyoulun on 9/8/2026.
//

#include "FullAttention.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenSession.h"
#include "tensor/GPUTensor.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

FullAttention::FullAttention(const FullAttnWeights &weights, const TextConfig &config)
    : weights_(weights), config_(config), type_index_(weights.type_index) {}

void FullAttention::prefill(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32) {
    const size_t input_size = static_cast<size_t>(g_hidden_f32.rows());
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;
    const int start_pos = kv.seq_len;

    // q_proj 输出 q_total*2（每 head 交错 [q, gate]）；k/v_proj 输出 kv_total。
    GPUTensor g_full_projection_f32 = GPUTensor(
        scratch, scratch_key::kFullProjection,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(q_total * 2)}, DType::F32);
    GPUTensor g_full_k_f32 = GPUTensor(
        scratch, scratch_key::kFullK,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(kv_total)}, DType::F32);
    GPUTensor g_full_v_f32 = GPUTensor(
        scratch, scratch_key::kFullV,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(kv_total)}, DType::F32);
    GPUTensor g_full_q_f32 = GPUTensor(
        scratch, scratch_key::kFullQ,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(q_total)}, DType::F32);
    GPUTensor g_full_gate_f32 = GPUTensor(
        scratch, scratch_key::kFullGate,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(q_total)}, DType::F32);
    GPUTensor g_full_attn_f32 = GPUTensor(
        scratch, scratch_key::kFullAttn,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(q_total)}, DType::F32);
    TensorTool::gemm(weights_.s_q_proj, g_hidden_f32, g_full_projection_f32, scratch, scratch_key::kInputLowp, "fullattn.d_q_proj");
    TensorTool::gemm(weights_.s_k_proj, g_hidden_f32, g_full_k_f32, scratch, scratch_key::kInputLowp, "fullattn.d_k_proj");
    TensorTool::gemm(weights_.s_v_proj, g_hidden_f32, g_full_v_f32, scratch, scratch_key::kInputLowp, "fullattn.d_v_proj");

    TensorTool::full_attention_q_batch(g_full_projection_f32, weights_.s_q_norm, g_full_q_f32, g_full_gate_f32,
                                  n_heads, head_dim, start_pos, theta, partial, eps);
    TensorTool::full_attention_kv_batch(g_full_k_f32, g_full_v_f32, weights_.s_k_norm, kv.g_key_cache_f32, kv.g_value_cache_f32,
                                   kv_heads, head_dim, /*max_seq_len=*/0, start_pos, theta, partial, eps);
    TensorTool::full_attention_attend_batch(g_full_q_f32, g_full_gate_f32, kv.g_key_cache_f32, kv.g_value_cache_f32, g_full_attn_f32,
                                       n_heads, kv_heads, head_dim, /*max_seq_len=*/0, start_pos);

    // o_proj：[g_hidden, q_total] · attn[q_total, tokens] -> [g_hidden, tokens]。
    TensorTool::gemm(weights_.s_o_proj, g_full_attn_f32, g_out_f32, scratch, scratch_key::kFullAttnLowp, "fullattn.d_o_proj");

    kv.seq_len += input_size;
    // 不做全设备同步：同流顺序保证依赖，barrier 交给前向末尾的 lm_head 同步 D2H。
    check_cuda(cudaGetLastError(), "FullAttention prefill kernel launch 失败");
}

void FullAttention::decode(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32, int pos) {
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;

    GPUTensor g_full_projection_f32 = GPUTensor(
        scratch, scratch_key::kFullProjection, {1, static_cast<int64_t>(q_total * 2)}, DType::F32);
    GPUTensor g_full_k_f32 = GPUTensor(
        scratch, scratch_key::kFullK, {1, static_cast<int64_t>(kv_total)}, DType::F32);
    GPUTensor g_full_v_f32 = GPUTensor(
        scratch, scratch_key::kFullV, {1, static_cast<int64_t>(kv_total)}, DType::F32);
    GPUTensor g_full_q_f32 = GPUTensor(
        scratch, scratch_key::kFullQ, {1, static_cast<int64_t>(q_total)}, DType::F32);
    GPUTensor g_full_gate_f32 = GPUTensor(
        scratch, scratch_key::kFullGate, {1, static_cast<int64_t>(q_total)}, DType::F32);
    GPUTensor g_full_attn_f32 = GPUTensor(
        scratch, scratch_key::kFullAttn, {1, static_cast<int64_t>(q_total)}, DType::F32);
    TensorTool::gemm(weights_.s_q_proj, g_hidden_f32, g_full_projection_f32, scratch, scratch_key::kInputLowp, "fullattn.d_q_proj");
    TensorTool::gemm(weights_.s_k_proj, g_hidden_f32, g_full_k_f32, scratch, scratch_key::kInputLowp, "fullattn.d_k_proj");
    TensorTool::gemm(weights_.s_v_proj, g_hidden_f32, g_full_v_f32, scratch, scratch_key::kInputLowp, "fullattn.d_v_proj");

    TensorTool::full_attention_q(g_full_projection_f32, weights_.s_q_norm, g_full_q_f32, g_full_gate_f32,
                            n_heads, head_dim, pos, theta, partial, eps);
    TensorTool::full_attention_kv(g_full_k_f32, g_full_v_f32, weights_.s_k_norm, kv.g_key_cache_f32, kv.g_value_cache_f32,
                             kv_heads, head_dim, /*max_seq_len=*/0, pos, theta, partial, eps);
    TensorTool::full_attention_attend(g_full_q_f32, g_full_gate_f32, kv.g_key_cache_f32, kv.g_value_cache_f32, g_full_attn_f32,
                                 n_heads, kv_heads, head_dim, /*max_seq_len=*/0, pos);

    TensorTool::gemm(weights_.s_o_proj, g_full_attn_f32, g_out_f32, scratch, scratch_key::kFullAttnLowp, "fullattn.d_o_proj");

    kv.seq_len = pos + 1;
    // 同 prefill：不做全设备同步，依赖同流顺序 + lm_head 末尾同步 D2H。
    check_cuda(cudaGetLastError(), "FullAttention decode kernel launch 失败");
}
