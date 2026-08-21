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
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

LinearAttention::LinearAttention(const LinearAttnWeights &weights, const TextConfig &config)
    : weights_(weights), config_(config), type_index_(weights.type_index) {}

void LinearAttention::prefill(QwenSession &session, const Tensor &hidden, const Tensor &out) {
    const size_t input_size = static_cast<size_t>(hidden.rows());
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

    float *d_linear_projection = scratch.ensure<float>(scratch_key::kLinearProjection, input_size * static_cast<size_t>(conv_dim));
    float *d_linear_z = scratch.ensure<float>(scratch_key::kLinearZ, input_size * static_cast<size_t>(value_total));
    float *d_linear_b = scratch.ensure<float>(scratch_key::kLinearB, input_size * static_cast<size_t>(value_heads));
    float *d_linear_a = scratch.ensure<float>(scratch_key::kLinearA, input_size * static_cast<size_t>(value_heads));
    float *d_linear_conv_out = scratch.ensure<float>(scratch_key::kLinearConvOut, input_size * static_cast<size_t>(conv_dim));
    float *d_linear_gated = scratch.ensure<float>(scratch_key::kLinearGated, input_size * static_cast<size_t>(value_total));

    Tensor linear_projection = Tensor::gpu_view(d_linear_projection, {static_cast<int64_t>(input_size), static_cast<int64_t>(conv_dim)});
    Tensor linear_z = Tensor::gpu_view(d_linear_z, {static_cast<int64_t>(input_size), static_cast<int64_t>(value_total)});
    Tensor linear_b = Tensor::gpu_view(d_linear_b, {static_cast<int64_t>(input_size), static_cast<int64_t>(value_heads)});
    Tensor linear_a = Tensor::gpu_view(d_linear_a, {static_cast<int64_t>(input_size), static_cast<int64_t>(value_heads)});
    Tensor linear_conv_out = Tensor::gpu_view(d_linear_conv_out, {static_cast<int64_t>(input_size), static_cast<int64_t>(conv_dim)});
    Tensor linear_gated = Tensor::gpu_view(d_linear_gated, {static_cast<int64_t>(input_size), static_cast<int64_t>(value_total)});
    TensorTool::gemm(weights_.in_proj_qkv, hidden, linear_projection, scratch, scratch_key::kInputLowp, "linattn.in_proj_qkv");
    TensorTool::gemm(weights_.in_proj_z, hidden, linear_z, scratch, scratch_key::kInputLowp, "linattn.in_proj_z");
    TensorTool::gemm(weights_.in_proj_b, hidden, linear_b, scratch, scratch_key::kInputLowp, "linattn.in_proj_b");
    TensorTool::gemm(weights_.in_proj_a, hidden, linear_a, scratch, scratch_key::kInputLowp, "linattn.in_proj_a");

    Tensor conv_state = Tensor::gpu_view(state.conv_state.ptr, {static_cast<int64_t>(conv_dim), static_cast<int64_t>(kernel)});
    Tensor recurrent_state = Tensor::gpu_view(state.recurrent_state.ptr,
        {static_cast<int64_t>(value_heads), static_cast<int64_t>(k_dim), static_cast<int64_t>(v_dim)});

    TensorTool::linear_attention_conv_batch(linear_projection, weights_.conv1d, conv_state, linear_conv_out, kernel);
    TensorTool::linear_attention_recurrent_batch(linear_conv_out, linear_z, linear_b, linear_a,
        weights_.a_log, weights_.dt_bias, weights_.norm, recurrent_state, linear_gated,
        key_heads, value_heads, k_dim, v_dim, eps);

    // out_proj：[hidden, value_total] · gated[value_total, tokens] -> [hidden, tokens]。
    TensorTool::gemm(weights_.out_proj, linear_gated, out, scratch, scratch_key::kLinearGatedLowp, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention prefill 同步失败");
}

void LinearAttention::decode(QwenSession &session, const Tensor &hidden, const Tensor &out) {
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

    float *d_linear_projection = scratch.ensure<float>(scratch_key::kLinearProjection, conv_dim);
    float *d_linear_z = scratch.ensure<float>(scratch_key::kLinearZ, value_total);
    float *d_linear_b = scratch.ensure<float>(scratch_key::kLinearB, value_heads);
    float *d_linear_a = scratch.ensure<float>(scratch_key::kLinearA, value_heads);
    float *d_linear_conv_out = scratch.ensure<float>(scratch_key::kLinearConvOut, conv_dim);
    float *d_linear_gated = scratch.ensure<float>(scratch_key::kLinearGated, value_total);

    Tensor linear_projection = Tensor::gpu_view(d_linear_projection, {1, static_cast<int64_t>(conv_dim)});
    Tensor linear_z = Tensor::gpu_view(d_linear_z, {1, static_cast<int64_t>(value_total)});
    Tensor linear_b = Tensor::gpu_view(d_linear_b, {1, static_cast<int64_t>(value_heads)});
    Tensor linear_a = Tensor::gpu_view(d_linear_a, {1, static_cast<int64_t>(value_heads)});
    Tensor linear_conv_out = Tensor::gpu_view(d_linear_conv_out, {1, static_cast<int64_t>(conv_dim)});
    Tensor linear_gated = Tensor::gpu_view(d_linear_gated, {1, static_cast<int64_t>(value_total)});
    TensorTool::gemm(weights_.in_proj_qkv, hidden, linear_projection, scratch, scratch_key::kInputLowp, "linattn.in_proj_qkv");
    TensorTool::gemm(weights_.in_proj_z, hidden, linear_z, scratch, scratch_key::kInputLowp, "linattn.in_proj_z");
    TensorTool::gemm(weights_.in_proj_b, hidden, linear_b, scratch, scratch_key::kInputLowp, "linattn.in_proj_b");
    TensorTool::gemm(weights_.in_proj_a, hidden, linear_a, scratch, scratch_key::kInputLowp, "linattn.in_proj_a");

    Tensor conv_state = Tensor::gpu_view(state.conv_state.ptr, {static_cast<int64_t>(conv_dim), static_cast<int64_t>(kernel)});
    Tensor recurrent_state = Tensor::gpu_view(state.recurrent_state.ptr,
        {static_cast<int64_t>(value_heads), static_cast<int64_t>(k_dim), static_cast<int64_t>(v_dim)});

    TensorTool::linear_attention_conv(linear_projection, weights_.conv1d, conv_state, linear_conv_out, kernel);
    TensorTool::linear_attention_recurrent(linear_conv_out, linear_z, linear_b, linear_a,
        weights_.a_log, weights_.dt_bias, weights_.norm, recurrent_state, linear_gated,
        key_heads, value_heads, k_dim, v_dim, eps);

    TensorTool::gemm(weights_.out_proj, linear_gated, out, scratch, scratch_key::kLinearGatedLowp, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention decode 同步失败");
}
