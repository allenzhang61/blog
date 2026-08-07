#include "Embedding.h"

#include "../kernels/cuda/cuda_ops.h"

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
    if (!device_out || !cuda_embedding_lookup_device_token(weight_, device_token_id.data, device_out)) {
        throw std::runtime_error("CUDA device token embedding lookup 失败。");
    }
    return {device_out, hidden_size, 0};
}

} // namespace llm_inference
