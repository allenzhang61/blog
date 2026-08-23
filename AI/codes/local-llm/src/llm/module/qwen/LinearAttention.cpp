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

void LinearAttention::prefill(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32) {
    const size_t input_size = static_cast<size_t>(g_hidden_f32.rows());
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

    GPUTensor g_linear_projection_f32 = GPUTensor(
        scratch, scratch_key::kLinearProjection,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_z_f32 = GPUTensor(
        scratch, scratch_key::kLinearZ,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_total)}, DType::F32);
    GPUTensor g_linear_b_f32 = GPUTensor(
        scratch, scratch_key::kLinearB,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_a_f32 = GPUTensor(
        scratch, scratch_key::kLinearA,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_conv_out_f32 = GPUTensor(
        scratch, scratch_key::kLinearConvOut,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_gated_f32 = GPUTensor(
        scratch, scratch_key::kLinearGated,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(value_total)}, DType::F32);
    // 4 个 in_proj 共享同一份 hidden 输入且权重都是 bf16：只转一次 bf16 复用，
    // 省掉原先每个 GEMM 各自一次 f32->bf16 拷贝（旧路径 decode 每层 4 次冗余转换）。
    const void *d_hidden_lowp = TensorTool::prepare_lowp_input(
        g_hidden_f32, weights_.s_in_proj_qkv.dtype, scratch, scratch_key::kInputLowp);
    const int rows = static_cast<int>(g_hidden_f32.rows());
    TensorTool::gemm_lowp(weights_.s_in_proj_qkv, d_hidden_lowp, rows, g_linear_projection_f32, "linattn.d_in_proj_qkv");
    TensorTool::gemm_lowp(weights_.s_in_proj_z, d_hidden_lowp, rows, g_linear_z_f32, "linattn.d_in_proj_z");
    TensorTool::gemm_lowp(weights_.s_in_proj_b, d_hidden_lowp, rows, g_linear_b_f32, "linattn.d_in_proj_b");
    TensorTool::gemm_lowp(weights_.s_in_proj_a, d_hidden_lowp, rows, g_linear_a_f32, "linattn.d_in_proj_a");

    TensorTool::linear_attention_conv_batch(g_linear_projection_f32, weights_.s_conv1d, state.g_conv_state_f32,
                                            g_linear_conv_out_f32, kernel);
    TensorTool::linear_attention_recurrent_batch(g_linear_conv_out_f32, g_linear_z_f32, g_linear_b_f32, g_linear_a_f32,
        weights_.s_a_log, weights_.s_dt_bias, weights_.s_norm, state.g_recurrent_state_f32, g_linear_gated_f32,
        key_heads, value_heads, k_dim, v_dim, eps);

    // out_proj：[g_hidden, value_total] · gated[value_total, tokens] -> [g_hidden, tokens]。
    TensorTool::gemm(weights_.s_out_proj, g_linear_gated_f32, g_out_f32, scratch, scratch_key::kLinearGatedLowp, "linattn.d_out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention prefill 同步失败");
}

void LinearAttention::decode(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32) {
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

    GPUTensor g_linear_projection_f32 = GPUTensor(
        scratch, scratch_key::kLinearProjection, {1, static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_z_f32 = GPUTensor(
        scratch, scratch_key::kLinearZ, {1, static_cast<int64_t>(value_total)}, DType::F32);
    GPUTensor g_linear_b_f32 = GPUTensor(
        scratch, scratch_key::kLinearB, {1, static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_a_f32 = GPUTensor(
        scratch, scratch_key::kLinearA, {1, static_cast<int64_t>(value_heads)}, DType::F32);
    GPUTensor g_linear_conv_out_f32 = GPUTensor(
        scratch, scratch_key::kLinearConvOut, {1, static_cast<int64_t>(conv_dim)}, DType::F32);
    GPUTensor g_linear_gated_f32 = GPUTensor(
        scratch, scratch_key::kLinearGated, {1, static_cast<int64_t>(value_total)}, DType::F32);
    // 同 prefill：4 个 in_proj 共享 hidden、权重都是 bf16，只转一次 bf16 复用。
    const void *d_hidden_lowp = TensorTool::prepare_lowp_input(
        g_hidden_f32, weights_.s_in_proj_qkv.dtype, scratch, scratch_key::kInputLowp);
    const int rows = static_cast<int>(g_hidden_f32.rows());
    TensorTool::gemm_lowp(weights_.s_in_proj_qkv, d_hidden_lowp, rows, g_linear_projection_f32, "linattn.d_in_proj_qkv");
    TensorTool::gemm_lowp(weights_.s_in_proj_z, d_hidden_lowp, rows, g_linear_z_f32, "linattn.d_in_proj_z");
    TensorTool::gemm_lowp(weights_.s_in_proj_b, d_hidden_lowp, rows, g_linear_b_f32, "linattn.d_in_proj_b");
    TensorTool::gemm_lowp(weights_.s_in_proj_a, d_hidden_lowp, rows, g_linear_a_f32, "linattn.d_in_proj_a");

    TensorTool::linear_attention_conv(g_linear_projection_f32, weights_.s_conv1d, state.g_conv_state_f32,
                                      g_linear_conv_out_f32, kernel);
    TensorTool::linear_attention_recurrent(g_linear_conv_out_f32, g_linear_z_f32, g_linear_b_f32, g_linear_a_f32,
        weights_.s_a_log, weights_.s_dt_bias, weights_.s_norm, state.g_recurrent_state_f32, g_linear_gated_f32,
        key_heads, value_heads, k_dim, v_dim, eps);

    TensorTool::gemm(weights_.s_out_proj, g_linear_gated_f32, g_out_f32, scratch, scratch_key::kLinearGatedLowp, "linattn.d_out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention decode 同步失败");
}
