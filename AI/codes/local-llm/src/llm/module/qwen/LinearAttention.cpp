//
// Created by zhangyoulun on 9/8/2026.
//

#include "LinearAttention.h"

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

LinearAttention::LinearAttention(const LinearAttnWeights &weights, const TextConfig &config)
    : weights_(weights), config_(config), type_index_(weights.type_index) {}

void LinearAttention::prefill(QwenSession &session, const GPUTensor &g_hidden, const GPUTensor &g_out) {
    const size_t input_size = static_cast<size_t>(g_hidden.rows());
    LinearAttnRecurrentState &state = session.linear_attn_recurrent_states[type_index_];
    CudaScratch &scratch = session.scratch;
    const int key_heads = config_.linear_num_key_heads;
    const int value_heads = config_.linear_num_value_heads;
    const int k_dim = config_.linear_key_head_dim;
    const int v_dim = config_.linear_value_head_dim;
    const int kernel = config_.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.rms_norm_eps;

    GPUTensor g_linear_projection = GPUTensor(
        scratch, scratch_key::kLinearProjection,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_z = GPUTensor(
        scratch, scratch_key::kLinearZ,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_total)}, DType::F32);
    GPUTensor g_linear_b = GPUTensor(
        scratch, scratch_key::kLinearB,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_a = GPUTensor(
        scratch, scratch_key::kLinearA,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_conv_out = GPUTensor(
        scratch, scratch_key::kLinearConvOut,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_gated = GPUTensor(
        scratch, scratch_key::kLinearGated,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_total)}, DType::F32);
    TensorTool::gemm(weights_.s_in_proj_qkv, g_hidden, g_linear_projection, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_qkv");
    TensorTool::gemm(weights_.s_in_proj_z, g_hidden, g_linear_z, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_z");
    TensorTool::gemm(weights_.s_in_proj_b, g_hidden, g_linear_b, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_b");
    TensorTool::gemm(weights_.s_in_proj_a, g_hidden, g_linear_a, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_a");

    TensorTool::linear_attention_conv_batch(g_linear_projection, weights_.s_conv1d, state.g_conv_state,
                                            g_linear_conv_out, kernel);
    TensorTool::linear_attention_recurrent_batch(g_linear_conv_out, g_linear_z, g_linear_b, g_linear_a,
        weights_.s_a_log, weights_.s_dt_bias, weights_.s_norm, state.g_recurrent_state, g_linear_gated,
        key_heads, value_heads, k_dim, v_dim, eps);

    // out_proj：[g_hidden, value_total] · gated[value_total, tokens] -> [g_hidden, tokens]。
    TensorTool::gemm(weights_.s_out_proj, g_linear_gated, g_out, scratch, scratch_key::kLinearGatedLowp, "linattn.d_out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention prefill 同步失败");
}

void LinearAttention::decode(QwenSession &session, const GPUTensor &g_hidden, const GPUTensor &g_out) {
    LinearAttnRecurrentState &state = session.linear_attn_recurrent_states[type_index_];
    CudaScratch &scratch = session.scratch;
    const int key_heads = config_.linear_num_key_heads;
    const int value_heads = config_.linear_num_value_heads;
    const int k_dim = config_.linear_key_head_dim;
    const int v_dim = config_.linear_value_head_dim;
    const int kernel = config_.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.rms_norm_eps;

    GPUTensor g_linear_projection = GPUTensor(
        scratch, scratch_key::kLinearProjection, {1, static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_z = GPUTensor(
        scratch, scratch_key::kLinearZ, {1, static_cast<int64_t>(value_total)}, DType::F32);
    GPUTensor g_linear_b = GPUTensor(
        scratch, scratch_key::kLinearB, {1, static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_a = GPUTensor(
        scratch, scratch_key::kLinearA, {1, static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_conv_out = GPUTensor(
        scratch, scratch_key::kLinearConvOut, {1, static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_gated = GPUTensor(
        scratch, scratch_key::kLinearGated, {1, static_cast<int64_t>(value_total)}, DType::F32);
    TensorTool::gemm(weights_.s_in_proj_qkv, g_hidden, g_linear_projection, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_qkv");
    TensorTool::gemm(weights_.s_in_proj_z, g_hidden, g_linear_z, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_z");
    TensorTool::gemm(weights_.s_in_proj_b, g_hidden, g_linear_b, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_b");
    TensorTool::gemm(weights_.s_in_proj_a, g_hidden, g_linear_a, scratch, scratch_key::kInputLowp, "linattn.d_in_proj_a");

    TensorTool::linear_attention_conv(g_linear_projection, weights_.s_conv1d, state.g_conv_state,
                                      g_linear_conv_out, kernel);
    TensorTool::linear_attention_recurrent(g_linear_conv_out, g_linear_z, g_linear_b, g_linear_a,
        weights_.s_a_log, weights_.s_dt_bias, weights_.s_norm, state.g_recurrent_state, g_linear_gated,
        key_heads, value_heads, k_dim, v_dim, eps);

    TensorTool::gemm(weights_.s_out_proj, g_linear_gated, g_out, scratch, scratch_key::kLinearGatedLowp, "linattn.d_out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention decode 同步失败");
}
