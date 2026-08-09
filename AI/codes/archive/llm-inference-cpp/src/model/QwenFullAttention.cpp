#include "QwenFullAttention.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../kernels/cuda/cache/CudaFullAttentionState.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"

#include <stdexcept>

namespace llm_inference {

QwenFullAttention::QwenFullAttention(const ModelConfig & config, const FullAttnWeights & weights, int layer_index)
    : config_(config), weights_(weights), layer_index_(layer_index) {
}

const char * QwenFullAttention::name() const {
    return "QwenFullAttention";
}

Tensor QwenFullAttention::forward(const Tensor & normed_x, RunState & state) const {
    const int pos = state.sequence_length();
    const int hidden_dim = config_.text.hidden_size;
    const int n_heads = config_.text.num_attention_heads;
    const int kv_heads = config_.text.num_key_value_heads;
    const int head_dim = config_.text.head_dim;
    const int max_seq_len = state.full_state(layer_index_).max_seq_len;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.text.rms_norm_eps;
    const float rope_theta = config_.text.rope_parameters.rope_theta;
    const float partial_rotary_factor = config_.text.rope_parameters.partial_rotary_factor;

    auto & cache = cuda_weight_cache();
    DeviceWeight * projection_device = cache.cached_concat_weight(
        weights_.q_proj.info->name + "\n" + weights_.k_proj.info->name + "\n" + weights_.v_proj.info->name,
        {weights_.q_proj, weights_.k_proj, weights_.v_proj});
    DeviceWeight * q_norm_device = cache.cached_weight(weights_.q_norm);
    DeviceWeight * k_norm_device = cache.cached_weight(weights_.k_norm);
    DeviceWeight * attn_out_device = cache.cached_weight(weights_.o_proj);
    if (!projection_device || !q_norm_device || !k_norm_device || !attn_out_device) {
        throw std::runtime_error("CUDA full attention 权重缓存失败，layer=" + std::to_string(layer_index_));
    }

    cache.mixer_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(float), "layer mixer buffer");

    CudaFullAttentionState * cuda_state =
        CudaFullAttentionState::ensure(state.full_state(layer_index_).cuda_state, n_heads, kv_heads, head_dim, max_seq_len);
    WeightMeta combined_info = *weights_.q_proj.info;
    combined_info.name = weights_.q_proj.info->name + "+k+v";
    combined_info.shape[0] = static_cast<int64_t>(q_total * 2 + kv_total * 2);
    WeightData combined_ref {&combined_info, nullptr};
    cuda_weight_matvec_to_device(cache, combined_ref, *projection_device, normed_x.data, projection_device->type, cuda_state->projection);
    const float * q_and_gate_ptr = cuda_state->projection;
    const float * k_ptr = cuda_state->projection + q_total * 2;
    const float * v_ptr = cuda_state->projection + q_total * 2 + kv_total;

    launch_full_attention_q(q_and_gate_ptr, static_cast<const uint16_t *>(q_norm_device->ptr), cuda_state->q, cuda_state->gate, n_heads, head_dim, pos, rope_theta, partial_rotary_factor, eps, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_q device 失败");
    launch_full_attention_kv(k_ptr, v_ptr, static_cast<const uint16_t *>(k_norm_device->ptr), cuda_state->key_cache, cuda_state->value_cache, kv_heads, head_dim, max_seq_len, pos, rope_theta, partial_rotary_factor, eps, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_kv device 失败");
    launch_full_attention_attend(cuda_state->q, cuda_state->gate, cuda_state->key_cache, cuda_state->value_cache, cuda_state->attn, n_heads, kv_heads, head_dim, max_seq_len, pos, nullptr);
    check_cuda(cudaGetLastError(), "launch_full_attention_attend device 失败");
    cuda_float_to_lowp(cuda_state->attn, cuda_state->attn_bf16, q_total, attn_out_device->type);
    check_cuda(cudaGetLastError(), "cuda_float_to_lowp full attn device 失败");
    cuda_weight_matvec_to_device(cache, weights_.o_proj, *attn_out_device, cuda_state->attn_bf16, attn_out_device->type, cache.mixer_buffer);
    return {cache.mixer_buffer, hidden_dim, -1};
}

} // namespace llm_inference
