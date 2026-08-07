#include "QwenMlp.h"

#include "../kernels/cuda/cuda_ops.h"
#include "../kernels/cuda/cuda_weight_cache.h"

#include <stdexcept>

namespace llm_inference {

QwenMlp::QwenMlp(const MlpWeights & weights, int layer_index)
    : weights_(weights), layer_index_(layer_index) {
}

const char * QwenMlp::name() const {
    return "QwenMlp";
}

Tensor QwenMlp::forward(const Tensor & device_x) const {
    const int hidden_dim = static_cast<int>(weights_.gate.info->shape[1]);
    auto & cache = cuda_weight_cache();
    cache.mlp_out_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(float), "layer mlp out buffer");
    if (!cuda_mlp_from_device_bf16_to_device(
            weights_.gate,
            weights_.up,
            weights_.down,
            static_cast<const uint16_t *>(device_x.data),
            cache.mlp_out_buffer)) {
        throw std::runtime_error("CUDA MLP 失败，layer=" + std::to_string(layer_index_));
    }
    return {cache.mlp_out_buffer, hidden_dim, -1};
}

} // namespace llm_inference

