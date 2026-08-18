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
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

LinearAttention::LinearAttention(const LinearAttnWeights &weights, const TextConfig &config)
    : weights_(weights), config_(config), type_index_(weights.type_index) {}

void LinearAttention::prefill(QwenSession &session, const Tensor &hidden, const Tensor &out) {
    const float *d_hidden = hidden.gpu_f32();
    float *d_out = out.gpu_f32();
    const size_t input_size = static_cast<size_t>(hidden.rows());
    LinearAttnRecurrentState &state = session.linear_attn_recurrent_states[type_index_];
    CudaScratch &scratch = session.scratch;
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

    float *d_linear_projection = scratch.ensure<float>(scratch_key::kLinearProjection, input_size * static_cast<size_t>(conv_dim));
    float *d_linear_z = scratch.ensure<float>(scratch_key::kLinearZ, input_size * static_cast<size_t>(value_total));
    float *d_linear_b = scratch.ensure<float>(scratch_key::kLinearB, input_size * static_cast<size_t>(value_heads));
    float *d_linear_a = scratch.ensure<float>(scratch_key::kLinearA, input_size * static_cast<size_t>(value_heads));
    float *d_linear_conv_out = scratch.ensure<float>(scratch_key::kLinearConvOut, input_size * static_cast<size_t>(conv_dim));
    float *d_linear_gated = scratch.ensure<float>(scratch_key::kLinearGated, input_size * static_cast<size_t>(value_total));
    uint16_t *d_linear_gated_lowp =
        scratch.ensure<uint16_t>(scratch_key::kLinearGatedLowp, input_size * static_cast<size_t>(value_total));

    // 输入激活转成权重 dtype（BF16/F16）后再做各投影。
    uint16_t *d_input_lowp =
        scratch.ensure<uint16_t>(scratch_key::kInputLowp, input_size * static_cast<size_t>(hidden_size));
    CudaWeight qkv = weights_.in_proj_qkv.cached_weight()->try_dequant();
    GemmInput in = prepare_gemm_input(d_hidden, d_input_lowp, input_size * static_cast<size_t>(hidden_size), qkv.type, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, qkv, in.ptr, d_linear_projection, conv_dim, hidden_size, input_size, in.type, "linattn.in_proj_qkv");

    CudaWeight in_proj_z = weights_.in_proj_z.cached_weight()->try_dequant();
    gemm_weight(global_cuda_weight_pool().handle, in_proj_z, in.ptr, d_linear_z, value_total, hidden_size, input_size, in.type, "linattn.in_proj_z");
    CudaWeight in_proj_b = weights_.in_proj_b.cached_weight()->try_dequant();
    gemm_weight(global_cuda_weight_pool().handle, in_proj_b, in.ptr, d_linear_b, value_heads, hidden_size, input_size, in.type, "linattn.in_proj_b");
    CudaWeight in_proj_a = weights_.in_proj_a.cached_weight()->try_dequant();
    gemm_weight(global_cuda_weight_pool().handle, in_proj_a, in.ptr, d_linear_a, value_heads, hidden_size, input_size, in.type, "linattn.in_proj_a");

    float *conv_state = static_cast<float *>(state.conv_state.ptr);
    float *recurrent_state = static_cast<float *>(state.recurrent_state.ptr);

    CudaWeight conv = weights_.conv1d.cached_weight()->try_dequant();
    launch_linear_attention_conv_batch(d_linear_projection, static_cast<const uint16_t *>(conv.ptr),
                                       conv_state, d_linear_conv_out, token_count, conv_dim, kernel, nullptr);
    CudaWeight a_log = weights_.a_log.cached_weight()->try_dequant();
    CudaWeight dt_bias = weights_.dt_bias.cached_weight()->try_dequant();
    CudaWeight norm = weights_.norm.cached_weight()->try_dequant();
    launch_linear_attention_recurrent_batch(
        d_linear_conv_out, d_linear_z, d_linear_b, d_linear_a, static_cast<const float *>(a_log.ptr),
        static_cast<const uint16_t *>(dt_bias.ptr), static_cast<const float *>(norm.ptr),
        recurrent_state, d_linear_gated, token_count, key_heads, value_heads, k_dim, v_dim, eps, nullptr);

    CudaWeight out_proj = weights_.out_proj.cached_weight()->try_dequant();
    GemmInput out_in = prepare_gemm_input(d_linear_gated, d_linear_gated_lowp, input_size * static_cast<size_t>(value_total), out_proj.type, nullptr);

    // out_proj：[hidden, value_total] · gated[value_total, tokens] -> [hidden, tokens]。
    gemm_weight(global_cuda_weight_pool().handle, out_proj, out_in.ptr, d_out, hidden_size, value_total, input_size, out_in.type, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention prefill 同步失败");
}

void LinearAttention::decode(QwenSession &session, const Tensor &hidden, const Tensor &out) {
    const float *d_hidden = hidden.gpu_f32();
    float *d_out = out.gpu_f32();
    LinearAttnRecurrentState &state = session.linear_attn_recurrent_states[type_index_];
    CudaScratch &scratch = session.scratch;
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

    float *d_linear_projection = scratch.ensure<float>(scratch_key::kLinearProjection, conv_dim);
    float *d_linear_z = scratch.ensure<float>(scratch_key::kLinearZ, value_total);
    float *d_linear_b = scratch.ensure<float>(scratch_key::kLinearB, value_heads);
    float *d_linear_a = scratch.ensure<float>(scratch_key::kLinearA, value_heads);
    float *d_linear_conv_out = scratch.ensure<float>(scratch_key::kLinearConvOut, conv_dim);
    float *d_linear_gated = scratch.ensure<float>(scratch_key::kLinearGated, value_total);
    uint16_t *d_linear_gated_lowp = scratch.ensure<uint16_t>(scratch_key::kLinearGatedLowp, value_total);

    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(scratch_key::kInputLowp, hidden_size);
    CudaWeight qkv = weights_.in_proj_qkv.cached_weight()->try_dequant();
    GemmInput in = prepare_gemm_input(d_hidden, d_input_lowp, hidden_size, qkv.type, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, qkv, in.ptr, d_linear_projection, conv_dim, hidden_size, 1, in.type, "linattn.in_proj_qkv");

    CudaWeight in_proj_z = weights_.in_proj_z.cached_weight()->try_dequant();
    gemm_weight(global_cuda_weight_pool().handle, in_proj_z, in.ptr, d_linear_z, value_total, hidden_size, 1, in.type, "linattn.in_proj_z");
    CudaWeight in_proj_b = weights_.in_proj_b.cached_weight()->try_dequant();
    gemm_weight(global_cuda_weight_pool().handle, in_proj_b, in.ptr, d_linear_b, value_heads, hidden_size, 1, in.type, "linattn.in_proj_b");
    CudaWeight in_proj_a = weights_.in_proj_a.cached_weight()->try_dequant();
    gemm_weight(global_cuda_weight_pool().handle, in_proj_a, in.ptr, d_linear_a, value_heads, hidden_size, 1, in.type, "linattn.in_proj_a");

    float *conv_state = static_cast<float *>(state.conv_state.ptr);
    float *recurrent_state = static_cast<float *>(state.recurrent_state.ptr);

    CudaWeight conv = weights_.conv1d.cached_weight()->try_dequant();
    launch_linear_attention_conv(d_linear_projection, static_cast<const uint16_t *>(conv.ptr),
                                 conv_state, d_linear_conv_out, conv_dim, kernel, nullptr);
    CudaWeight a_log = weights_.a_log.cached_weight()->try_dequant();
    CudaWeight dt_bias = weights_.dt_bias.cached_weight()->try_dequant();
    CudaWeight norm = weights_.norm.cached_weight()->try_dequant();
    launch_linear_attention_recurrent(
        d_linear_conv_out, d_linear_z, d_linear_b, d_linear_a, static_cast<const float *>(a_log.ptr),
        static_cast<const uint16_t *>(dt_bias.ptr), static_cast<const float *>(norm.ptr),
        recurrent_state, d_linear_gated, key_heads, value_heads, k_dim, v_dim, eps, nullptr);

    CudaWeight out_proj = weights_.out_proj.cached_weight()->try_dequant();
    GemmInput out_in = prepare_gemm_input(d_linear_gated, d_linear_gated_lowp, value_total, out_proj.type, nullptr);

    gemm_weight(global_cuda_weight_pool().handle, out_proj, out_in.ptr, d_out, hidden_size, value_total, 1, out_in.type, "linattn.out_proj");

    check_cuda(cudaDeviceSynchronize(), "LinearAttention decode 同步失败");
}
