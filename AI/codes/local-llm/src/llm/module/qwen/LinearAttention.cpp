//
// Created by zhangyoulun on 9/8/2026.
//

#include "LinearAttention.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenForwardScratch.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

LinearAttention::LinearAttention(const LinearAttnWeights &weights, const TextConfig &config,
                                 CudaWeightPool *pool)
    : weights_(weights), config_(config), pool_(pool) {}

void LinearAttention::prefill(const float *d_hidden, float *d_out, size_t input_size,
                              LinearAttnRecurrentState &state, QwenForwardScratch &scratch) {
    const int hidden_size = config_.hidden_size;
    const int key_heads = config_.linear_num_key_heads;
    const int value_heads = config_.linear_num_value_heads;
    const int k_dim = config_.linear_key_head_dim;
    const int v_dim = config_.linear_value_head_dim;
    const int kernel = config_.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.rms_norm_eps;
    const int token_count = static_cast<int>(input_size);

    float *d_mixed = scratch.linear_projection.ensure(input_size * static_cast<size_t>(conv_dim), "lin mixed");
    float *d_z = scratch.linear_z.ensure(input_size * static_cast<size_t>(value_total), "lin z");
    float *d_b = scratch.linear_b.ensure(input_size * static_cast<size_t>(value_heads), "lin b");
    float *d_a = scratch.linear_a.ensure(input_size * static_cast<size_t>(value_heads), "lin a");
    float *d_conv = scratch.linear_conv_out.ensure(input_size * static_cast<size_t>(conv_dim), "lin conv");
    float *d_gated = scratch.linear_gated.ensure(input_size * static_cast<size_t>(value_total), "lin gated");
    uint16_t *d_gated_lowp =
        scratch.linear_gated_lowp.ensure(input_size * static_cast<size_t>(value_total), "lin gated lowp");

    // 输入激活转成权重 dtype（BF16/F16）后再做各投影。
    uint16_t *d_in_lowp =
        scratch.input_lowp_buffer.ensure(input_size * static_cast<size_t>(hidden_size), "lin in lowp");
    CudaWeight qkv = pool_->cached_weight(weights_.in_proj_qkv)->try_dequant();
    GemmInput in = prepare_gemm_input(d_hidden, d_in_lowp, input_size * static_cast<size_t>(hidden_size), qkv.type, nullptr);
    gemm_weight(pool_->handle, qkv, in.ptr, d_mixed, conv_dim, hidden_size, input_size, in.type, "linattn.in_proj_qkv");

    CudaWeight z_w = pool_->cached_weight(weights_.in_proj_z)->try_dequant();
    gemm_weight(pool_->handle, z_w, in.ptr, d_z, value_total, hidden_size, input_size, in.type, "linattn.in_proj_z");
    CudaWeight b_w = pool_->cached_weight(weights_.in_proj_b)->try_dequant();
    gemm_weight(pool_->handle, b_w, in.ptr, d_b, value_heads, hidden_size, input_size, in.type, "linattn.in_proj_b");
    CudaWeight a_w = pool_->cached_weight(weights_.in_proj_a)->try_dequant();
    gemm_weight(pool_->handle, a_w, in.ptr, d_a, value_heads, hidden_size, input_size, in.type, "linattn.in_proj_a");

    float *conv_state = static_cast<float *>(state.conv_state.ptr);
    float *recurrent_state = static_cast<float *>(state.recurrent_state.ptr);

    CudaWeight conv = pool_->cached_weight(weights_.conv1d)->try_dequant();
    launch_linear_attention_conv_batch(d_mixed, static_cast<const uint16_t *>(conv.ptr),
                                       conv_state, d_conv, token_count, conv_dim, kernel, nullptr);
    CudaWeight a_log = pool_->cached_weight(weights_.a_log)->try_dequant();
    CudaWeight dt_bias = pool_->cached_weight(weights_.dt_bias)->try_dequant();
    CudaWeight norm = pool_->cached_weight(weights_.norm)->try_dequant();
    launch_linear_attention_recurrent_batch(
        d_conv, d_z, d_b, d_a, static_cast<const float *>(a_log.ptr),
        static_cast<const uint16_t *>(dt_bias.ptr), static_cast<const float *>(norm.ptr),
        recurrent_state, d_gated, token_count, key_heads, value_heads, k_dim, v_dim, eps, nullptr);

    CudaWeight out_proj = pool_->cached_weight(weights_.out_proj)->try_dequant();
    GemmInput out_in = prepare_gemm_input(d_gated, d_gated_lowp, input_size * static_cast<size_t>(value_total), out_proj.type, nullptr);

    // out_proj：[hidden, value_total] · gated[value_total, tokens] -> [hidden, tokens]。
    gemm_weight(pool_->handle, out_proj, out_in.ptr, d_out, hidden_size, value_total, input_size, out_in.type, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention prefill 同步失败");
}

void LinearAttention::decode(const float *d_hidden, float *d_out,
                             LinearAttnRecurrentState &state, QwenForwardScratch &scratch) {
    const int hidden_size = config_.hidden_size;
    const int key_heads = config_.linear_num_key_heads;
    const int value_heads = config_.linear_num_value_heads;
    const int k_dim = config_.linear_key_head_dim;
    const int v_dim = config_.linear_value_head_dim;
    const int kernel = config_.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.rms_norm_eps;

    float *d_mixed = scratch.linear_projection.ensure(conv_dim, "lin mixed");
    float *d_z = scratch.linear_z.ensure(value_total, "lin z");
    float *d_b = scratch.linear_b.ensure(value_heads, "lin b");
    float *d_a = scratch.linear_a.ensure(value_heads, "lin a");
    float *d_conv = scratch.linear_conv_out.ensure(conv_dim, "lin conv");
    float *d_gated = scratch.linear_gated.ensure(value_total, "lin gated");
    uint16_t *d_gated_lowp = scratch.linear_gated_lowp.ensure(value_total, "lin gated lowp");

    uint16_t *d_in_lowp = scratch.input_lowp_buffer.ensure(hidden_size, "lin in lowp");
    CudaWeight qkv = pool_->cached_weight(weights_.in_proj_qkv)->try_dequant();
    GemmInput in = prepare_gemm_input(d_hidden, d_in_lowp, hidden_size, qkv.type, nullptr);
    gemm_weight(pool_->handle, qkv, in.ptr, d_mixed, conv_dim, hidden_size, 1, in.type, "linattn.in_proj_qkv");

    CudaWeight z_w = pool_->cached_weight(weights_.in_proj_z)->try_dequant();
    gemm_weight(pool_->handle, z_w, in.ptr, d_z, value_total, hidden_size, 1, in.type, "linattn.in_proj_z");
    CudaWeight b_w = pool_->cached_weight(weights_.in_proj_b)->try_dequant();
    gemm_weight(pool_->handle, b_w, in.ptr, d_b, value_heads, hidden_size, 1, in.type, "linattn.in_proj_b");
    CudaWeight a_w = pool_->cached_weight(weights_.in_proj_a)->try_dequant();
    gemm_weight(pool_->handle, a_w, in.ptr, d_a, value_heads, hidden_size, 1, in.type, "linattn.in_proj_a");

    float *conv_state = static_cast<float *>(state.conv_state.ptr);
    float *recurrent_state = static_cast<float *>(state.recurrent_state.ptr);

    CudaWeight conv = pool_->cached_weight(weights_.conv1d)->try_dequant();
    launch_linear_attention_conv(d_mixed, static_cast<const uint16_t *>(conv.ptr),
                                 conv_state, d_conv, conv_dim, kernel, nullptr);
    CudaWeight a_log = pool_->cached_weight(weights_.a_log)->try_dequant();
    CudaWeight dt_bias = pool_->cached_weight(weights_.dt_bias)->try_dequant();
    CudaWeight norm = pool_->cached_weight(weights_.norm)->try_dequant();
    launch_linear_attention_recurrent(
        d_conv, d_z, d_b, d_a, static_cast<const float *>(a_log.ptr),
        static_cast<const uint16_t *>(dt_bias.ptr), static_cast<const float *>(norm.ptr),
        recurrent_state, d_gated, key_heads, value_heads, k_dim, v_dim, eps, nullptr);

    CudaWeight out_proj = pool_->cached_weight(weights_.out_proj)->try_dequant();
    GemmInput out_in = prepare_gemm_input(d_gated, d_gated_lowp, value_total, out_proj.type, nullptr);

    gemm_weight(pool_->handle, out_proj, out_in.ptr, d_out, hidden_size, value_total, 1, out_in.type, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention decode 同步失败");
}
