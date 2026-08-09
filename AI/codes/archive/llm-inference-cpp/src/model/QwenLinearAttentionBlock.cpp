#include "QwenLinearAttentionBlock.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"

#include <stdexcept>

namespace llm_inference {

QwenLinearAttentionBlock::QwenLinearAttentionBlock(const ModelConfig & config, const LayerWeights & weights, int layer_index)
    : QwenBlock(config, weights, layer_index),
      linear_attention_(config, weights.lin, layer_index),
      mlp_(weights.mlp, layer_index) {
}

const char * QwenLinearAttentionBlock::name() const {
    return "QwenLinearAttentionBlock";
}

Tensor QwenLinearAttentionBlock::forward(const Tensor & device_x, RunState & state) const {
    const Tensor output = allocate_output(device_x);
    if (weights_.input_norm.info->dtype != "BF16") {
        throw std::runtime_error("CUDA linear attention block 失败，layer=" + std::to_string(layer_index_));
    }
    const int hidden_dim = config_.text.hidden_size;
    const float eps = config_.text.rms_norm_eps;

    auto & cache = cuda_weight_cache();
    DeviceWeight * input_norm_device = cache.cached_weight(weights_.input_norm);
    DeviceWeight * post_norm_device = cache.cached_weight(weights_.post_norm);
    DeviceWeight * projection_device = cache.cached_concat_weight(
        weights_.lin.in_proj_qkv.info->name + "\n" + weights_.lin.in_proj_z.info->name + "\n" + weights_.lin.in_proj_b.info->name + "\n" + weights_.lin.in_proj_a.info->name,
        {weights_.lin.in_proj_qkv, weights_.lin.in_proj_z, weights_.lin.in_proj_b, weights_.lin.in_proj_a});
    if (!input_norm_device || !post_norm_device || !projection_device) {
        throw std::runtime_error("CUDA linear attention block 权重缓存失败，layer=" + std::to_string(layer_index_));
    }

    cache.layer_out_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(float), "layer out buffer");
    cache.norm_lowp_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(uint16_t), "input norm bf16 buffer");
    cache.post_norm_lowp_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(uint16_t), "post norm bf16 buffer");

    if (projection_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_x.data),
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_lowp_buffer,
            hidden_dim,
            eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 device linear input 失败");
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_x.data),
            static_cast<const uint16_t *>(input_norm_device->ptr),
            cache.norm_lowp_buffer,
            hidden_dim,
            eps,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 device linear input 失败");
    }

    const Tensor normed {cache.norm_lowp_buffer, hidden_dim, -1};
    const Tensor mixer = linear_attention_.forward(normed, state);

    launch_add_rms_norm_to_bf16(
        static_cast<const float *>(device_x.data),
        static_cast<const float *>(mixer.data),
        static_cast<const uint16_t *>(post_norm_device->ptr),
        cache.layer_out_buffer,
        cache.post_norm_lowp_buffer,
        hidden_dim,
        eps,
        true,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_add_rms_norm_to_bf16 linear device post 失败");
    const Tensor post_normed {cache.post_norm_lowp_buffer, hidden_dim, -1};
    const Tensor mlp_out = mlp_.forward(post_normed);
    launch_add_float(cache.layer_out_buffer, static_cast<const float *>(mlp_out.data), static_cast<float *>(output.data), hidden_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_add_float linear device mlp residual 失败");
    return output;
}

} // namespace llm_inference
