#include "QwenLinearAttention.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../kernels/cuda/cache/CudaLinearAttentionState.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"

#include <stdexcept>

namespace llm_inference {

QwenLinearAttention::QwenLinearAttention(const ModelConfig & config, const LinearAttnWeights & weights, int layer_index)
    : config_(config), weights_(weights), layer_index_(layer_index) {
}

const char * QwenLinearAttention::name() const {
    return "QwenLinearAttention";
}

Tensor QwenLinearAttention::forward(const Tensor & normed_x, RunState & state) const {
    const int hidden_dim = config_.text.hidden_size;
    const int key_heads = config_.text.linear_num_key_heads;
    const int value_heads = config_.text.linear_num_value_heads;
    const int k_dim = config_.text.linear_key_head_dim;
    const int v_dim = config_.text.linear_value_head_dim;
    const int kernel = config_.text.linear_conv_kernel_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const float eps = config_.text.rms_norm_eps;

    auto & cache = cuda_weight_cache();
    DeviceWeight * projection_device = cache.cached_concat_weight(
        weights_.in_proj_qkv.info->name + "\n" + weights_.in_proj_z.info->name + "\n" + weights_.in_proj_b.info->name + "\n" + weights_.in_proj_a.info->name,
        {weights_.in_proj_qkv, weights_.in_proj_z, weights_.in_proj_b, weights_.in_proj_a});
    DeviceWeight * conv_device = cache.cached_weight(weights_.conv1d);
    DeviceWeight * a_log_device = cache.cached_weight(weights_.a_log);
    DeviceWeight * dt_bias_device = cache.cached_weight(weights_.dt_bias);
    DeviceWeight * attn_norm_device = cache.cached_weight(weights_.norm);
    DeviceWeight * attn_out_device = cache.cached_weight(weights_.out_proj);
    if (!projection_device || !conv_device || !a_log_device || !dt_bias_device || !attn_norm_device || !attn_out_device) {
        throw std::runtime_error("CUDA linear attention 权重缓存失败，layer=" + std::to_string(layer_index_));
    }

    cache.mixer_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(float), "layer mixer buffer");
    CudaLinearAttentionState * cuda_state =
        CudaLinearAttentionState::ensure(state.linear[layer_index_].cuda_state, key_heads, value_heads, k_dim, v_dim, kernel);

    WeightMeta combined_info = *weights_.in_proj_qkv.info;
    combined_info.name = weights_.in_proj_qkv.info->name + "+z+b+a";
    combined_info.shape[0] = static_cast<int64_t>(conv_dim + value_total + value_heads * 2);
    WeightData combined_ref {&combined_info, nullptr};
    cuda_weight_matvec_to_device(cache, combined_ref, *projection_device, normed_x.data, projection_device->type, cuda_state->projection);

    const float * mixed_ptr = cuda_state->projection;
    const float * z_ptr = cuda_state->projection + conv_dim;
    const float * b_ptr = cuda_state->projection + conv_dim + value_total;
    const float * a_ptr = cuda_state->projection + conv_dim + value_total + value_heads;

    launch_linear_attention_conv(mixed_ptr, static_cast<const uint16_t *>(conv_device->ptr), cuda_state->conv_state, cuda_state->conv_out, conv_dim, kernel, nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_conv device 失败");
    launch_linear_attention_recurrent(
        cuda_state->conv_out,
        z_ptr,
        b_ptr,
        a_ptr,
        static_cast<const float *>(a_log_device->ptr),
        static_cast<const uint16_t *>(dt_bias_device->ptr),
        static_cast<const float *>(attn_norm_device->ptr),
        cuda_state->recurrent_state,
        cuda_state->gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_linear_attention_recurrent device 失败");
    cuda_float_to_lowp(cuda_state->gated, cuda_state->gated_bf16, value_total, attn_out_device->type);
    check_cuda(cudaGetLastError(), "cuda_float_to_lowp linear gated device 失败");
    cuda_weight_matvec_to_device(cache, weights_.out_proj, *attn_out_device, cuda_state->gated_bf16, attn_out_device->type, cache.mixer_buffer);
    return {cache.mixer_buffer, hidden_dim, -1};
}

} // namespace llm_inference
