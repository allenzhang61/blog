//
// Created by zhangyoulun on 9/8/2026.
//

#include "LinearAttention.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "llm/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

LinearAttention::LinearAttention(const LinearAttnWeights &weights, const TextConfig &config,
                                 CudaWeightPool *pool)
    : weights_(weights), config_(config), pool_(pool) {}

void LinearAttention::prefill(const float *d_hidden, float *d_out, int tokens,
                              LinearAttnRecurrentState &state, QwenForwardScratch &scratch) {
    const int hidden = config_.hidden_size;
    const int key_heads = config_.linear_num_key_heads;
    const int value_heads = config_.linear_num_value_heads;
    const int k_dim = config_.linear_key_head_dim;
    const int v_dim = config_.linear_value_head_dim;
    const int kernel = config_.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.rms_norm_eps;

    CudaWeight *qkv = pool_->cached_weight(weights_.in_proj_qkv);
    CudaWeight *z_w = pool_->cached_weight(weights_.in_proj_z);
    CudaWeight *b_w = pool_->cached_weight(weights_.in_proj_b);
    CudaWeight *a_w = pool_->cached_weight(weights_.in_proj_a);
    CudaWeight *conv = pool_->cached_weight(weights_.conv1d);
    CudaWeight *a_log = pool_->cached_weight(weights_.a_log);
    CudaWeight *dt_bias = pool_->cached_weight(weights_.dt_bias);
    CudaWeight *norm = pool_->cached_weight(weights_.norm);
    CudaWeight *out_proj = pool_->cached_weight(weights_.out_proj);
    if (!qkv || !z_w || !b_w || !a_w || !conv || !a_log || !dt_bias || !norm || !out_proj) {
        throw std::runtime_error("LinearAttention 权重上传失败");
    }

    float *d_mixed = scratch.linear_projection.ensure(static_cast<size_t>(tokens) * conv_dim, "lin mixed");
    float *d_z = scratch.linear_z.ensure(static_cast<size_t>(tokens) * value_total, "lin z");
    float *d_b = scratch.linear_b.ensure(static_cast<size_t>(tokens) * value_heads, "lin b");
    float *d_a = scratch.linear_a.ensure(static_cast<size_t>(tokens) * value_heads, "lin a");
    float *d_conv = scratch.linear_conv_out.ensure(static_cast<size_t>(tokens) * conv_dim, "lin conv");
    float *d_gated = scratch.linear_gated.ensure(static_cast<size_t>(tokens) * value_total, "lin gated");
    uint16_t *d_gated_lowp =
        scratch.linear_gated_lowp.ensure(static_cast<size_t>(tokens) * value_total, "lin gated lowp");

    // 输入激活转成权重 dtype（BF16/F16）后再做各投影。
    uint16_t *d_in_lowp =
        scratch.input_lowp_buffer.ensure(static_cast<size_t>(tokens) * hidden, "lin in lowp");
    to_weight_lowp(d_hidden, d_in_lowp, tokens * hidden, *qkv, nullptr);

    gemm_weight(pool_->handle, *qkv, conv_dim, hidden, d_in_lowp, qkv->type, tokens, d_mixed, "linattn.in_proj_qkv");
    gemm_weight(pool_->handle, *z_w, value_total, hidden, d_in_lowp, z_w->type, tokens, d_z, "linattn.in_proj_z");
    gemm_weight(pool_->handle, *b_w, value_heads, hidden, d_in_lowp, b_w->type, tokens, d_b, "linattn.in_proj_b");
    gemm_weight(pool_->handle, *a_w, value_heads, hidden, d_in_lowp, a_w->type, tokens, d_a, "linattn.in_proj_a");

    float *conv_state = static_cast<float *>(state.conv_state.ptr);
    float *recurrent_state = static_cast<float *>(state.recurrent_state.ptr);

    launch_linear_attention_conv_batch(d_mixed, static_cast<const uint16_t *>(conv->ptr),
                                       conv_state, d_conv, tokens, conv_dim, kernel, nullptr);
    launch_linear_attention_recurrent_batch(
        d_conv, d_z, d_b, d_a, static_cast<const float *>(a_log->ptr),
        static_cast<const uint16_t *>(dt_bias->ptr), static_cast<const float *>(norm->ptr),
        recurrent_state, d_gated, tokens, key_heads, value_heads, k_dim, v_dim, eps, nullptr);

    to_weight_lowp(d_gated, d_gated_lowp, tokens * value_total, *out_proj, nullptr);

    // out_proj：[hidden, value_total] · gated[value_total, tokens] -> [hidden, tokens]。
    gemm_weight(pool_->handle, *out_proj, hidden, value_total, d_gated_lowp, out_proj->type, tokens, d_out, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention prefill 同步失败");
}

void LinearAttention::decode(const float *d_hidden, float *d_out,
                             LinearAttnRecurrentState &state, QwenForwardScratch &scratch) {
    const int hidden = config_.hidden_size;
    const int key_heads = config_.linear_num_key_heads;
    const int value_heads = config_.linear_num_value_heads;
    const int k_dim = config_.linear_key_head_dim;
    const int v_dim = config_.linear_value_head_dim;
    const int kernel = config_.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.rms_norm_eps;

    CudaWeight *qkv = pool_->cached_weight(weights_.in_proj_qkv);
    CudaWeight *z_w = pool_->cached_weight(weights_.in_proj_z);
    CudaWeight *b_w = pool_->cached_weight(weights_.in_proj_b);
    CudaWeight *a_w = pool_->cached_weight(weights_.in_proj_a);
    CudaWeight *conv = pool_->cached_weight(weights_.conv1d);
    CudaWeight *a_log = pool_->cached_weight(weights_.a_log);
    CudaWeight *dt_bias = pool_->cached_weight(weights_.dt_bias);
    CudaWeight *norm = pool_->cached_weight(weights_.norm);
    CudaWeight *out_proj = pool_->cached_weight(weights_.out_proj);
    if (!qkv || !z_w || !b_w || !a_w || !conv || !a_log || !dt_bias || !norm || !out_proj) {
        throw std::runtime_error("LinearAttention 权重上传失败");
    }

    float *d_mixed = scratch.linear_projection.ensure(conv_dim, "lin mixed");
    float *d_z = scratch.linear_z.ensure(value_total, "lin z");
    float *d_b = scratch.linear_b.ensure(value_heads, "lin b");
    float *d_a = scratch.linear_a.ensure(value_heads, "lin a");
    float *d_conv = scratch.linear_conv_out.ensure(conv_dim, "lin conv");
    float *d_gated = scratch.linear_gated.ensure(value_total, "lin gated");
    uint16_t *d_gated_lowp = scratch.linear_gated_lowp.ensure(value_total, "lin gated lowp");

    uint16_t *d_in_lowp = scratch.input_lowp_buffer.ensure(hidden, "lin in lowp");
    to_weight_lowp(d_hidden, d_in_lowp, hidden, *qkv, nullptr);

    gemm_weight(pool_->handle, *qkv, conv_dim, hidden, d_in_lowp, qkv->type, 1, d_mixed, "linattn.in_proj_qkv");
    gemm_weight(pool_->handle, *z_w, value_total, hidden, d_in_lowp, z_w->type, 1, d_z, "linattn.in_proj_z");
    gemm_weight(pool_->handle, *b_w, value_heads, hidden, d_in_lowp, b_w->type, 1, d_b, "linattn.in_proj_b");
    gemm_weight(pool_->handle, *a_w, value_heads, hidden, d_in_lowp, a_w->type, 1, d_a, "linattn.in_proj_a");

    float *conv_state = static_cast<float *>(state.conv_state.ptr);
    float *recurrent_state = static_cast<float *>(state.recurrent_state.ptr);

    launch_linear_attention_conv(d_mixed, static_cast<const uint16_t *>(conv->ptr),
                                 conv_state, d_conv, conv_dim, kernel, nullptr);
    launch_linear_attention_recurrent(
        d_conv, d_z, d_b, d_a, static_cast<const float *>(a_log->ptr),
        static_cast<const uint16_t *>(dt_bias->ptr), static_cast<const float *>(norm->ptr),
        recurrent_state, d_gated, key_heads, value_heads, k_dim, v_dim, eps, nullptr);

    to_weight_lowp(d_gated, d_gated_lowp, value_total, *out_proj, nullptr);

    gemm_weight(pool_->handle, *out_proj, hidden, value_total, d_gated_lowp, out_proj->type, 1, d_out, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention decode 同步失败");
}
