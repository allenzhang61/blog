#include "Embedding.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cuda_ops.h"

#include <cstdint>
#include <stdexcept>

namespace llm_inference {

TokenEmbedding::TokenEmbedding(const WeightData & weight)
    : weight_(weight) {
}

const char * TokenEmbedding::name() const {
    return "TokenEmbedding";
}

Tensor TokenEmbedding::forward(const Tensor & device_token_id) const {
    const int hidden_size = static_cast<int>(weight_.info->shape[1]);
    void * device_out = cuda_token_hidden_buffer(0, hidden_size);
    if (weight_.info->shape.size() != 2 || !device_token_id.data || !device_out) {
        throw std::runtime_error("CUDA device token embedding lookup 失败。");
    }
    DeviceWeight * device_weight = cuda_weight_cache().cached_weight(weight_);
    if (!device_weight || (device_weight->type != CUDA_R_16BF && device_weight->type != CUDA_R_16F)) {
        throw std::runtime_error("CUDA device token embedding lookup 失败。");
    }
    launch_lowp_embedding_id_to_float(
        static_cast<const uint16_t *>(device_weight->ptr),
        static_cast<const int *>(device_token_id.data),
        static_cast<float *>(device_out),
        static_cast<int>(weight_.info->shape[0]),
        hidden_size,
        device_weight->type == CUDA_R_16F ? 1 : 0,
        nullptr);
    check_cuda(cudaGetLastError(), "launch_lowp_embedding_id_to_float 失败");
    return {device_out, hidden_size, 0};
}

} // namespace llm_inference
