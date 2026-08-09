#include "QwenLmHead.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cuda_ops.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llm_inference {

QwenLmHead::QwenLmHead(const WeightData & final_norm, const WeightData & embedding, float rms_norm_eps)
    : final_norm_(final_norm),
      embedding_(embedding),
      hidden_size_(static_cast<int>(embedding.info->shape[1])),
      rms_norm_eps_(rms_norm_eps) {
}

const char * QwenLmHead::name() const {
    return "QwenLmHead";
}

void QwenLmHead::forward(const Tensor & device_hidden, const Tensor & device_token_out) const {
    if (!device_hidden.data ||
        !device_token_out.data ||
        final_norm_.info->dtype != "BF16" ||
        final_norm_.info->shape.size() != 1 ||
        final_norm_.info->shape[0] != hidden_size_ ||
        embedding_.info->shape.size() != 2 ||
        embedding_.info->shape[1] != hidden_size_) {
        throw std::runtime_error("CUDA final norm + argmax 到设备失败。");
    }

    auto & cache = cuda_weight_cache();
    DeviceWeight * emb_device = cache.cached_weight(embedding_);
    DeviceWeight * norm_device = cache.cached_weight(final_norm_);
    if (!norm_device || !emb_device || (emb_device->type != CUDA_R_16BF && emb_device->type != CUDA_R_16F)) {
        throw std::runtime_error("CUDA final norm + argmax 到设备失败。");
    }
    const int vocab = static_cast<int>(embedding_.info->shape[0]);
    cache.norm_lowp_buffer.ensure_bytes(static_cast<size_t>(hidden_size_) * sizeof(uint16_t), "final norm lowp");
    cache.y_buffer.ensure_bytes(static_cast<size_t>(vocab) * sizeof(float), "final logits");
    if (emb_device->type == CUDA_R_16F) {
        launch_rms_norm_to_f16(
            static_cast<const float *>(device_hidden.data),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_lowp_buffer,
            hidden_size_,
            rms_norm_eps_,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_f16 final device 失败");
        cuda_weight_matvec_to_device(cache, embedding_, *emb_device, cache.norm_lowp_buffer, CUDA_R_16F, cache.y_buffer);
    } else {
        launch_rms_norm_to_bf16(
            static_cast<const float *>(device_hidden.data),
            static_cast<const uint16_t *>(norm_device->ptr),
            cache.norm_lowp_buffer,
            hidden_size_,
            rms_norm_eps_,
            true,
            nullptr);
        check_cuda(cudaGetLastError(), "launch_rms_norm_to_bf16 final device 失败");
        cuda_weight_matvec_to_device(cache, embedding_, *emb_device, cache.norm_lowp_buffer, CUDA_R_16BF, cache.y_buffer);
    }

    const int blocks = (vocab + 255) / 256;
    cache.argmax_block_values.ensure_bytes(static_cast<size_t>(blocks) * sizeof(float), "argmax block values");
    cache.argmax_block_indices.ensure_bytes(static_cast<size_t>(blocks) * sizeof(int), "argmax block indices");
    cache.argmax_best_value.ensure_bytes(sizeof(float), "argmax best value");
    cache.argmax_best_index.ensure_bytes(sizeof(int), "argmax best index");
    launch_argmax_float(
        cache.y_buffer,
        vocab,
        cache.argmax_block_values,
        cache.argmax_block_indices,
        cache.argmax_best_value,
        cache.argmax_best_index,
        blocks,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_argmax_float final device 失败");
    launch_copy_int(cache.argmax_best_index, static_cast<int *>(device_token_out.data), nullptr);
    check_cuda(cudaGetLastError(), "launch_copy_int final token 失败");
}

} // namespace llm_inference
