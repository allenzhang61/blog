#include "cpu_ops.h"

#include <stdexcept>
#include <string>

namespace llm_inference {
namespace cpu {

void embedding_lookup(const TensorRef & emb, int token_id, std::vector<float> & y) {
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden = static_cast<int>(emb.info->shape[1]);
    if (token_id < 0 || token_id >= vocab) {
        throw std::runtime_error("token id 越界：" + std::to_string(token_id));
    }
    y.resize(hidden);
    const size_t base = static_cast<size_t>(token_id) * static_cast<size_t>(hidden);
    for (int i = 0; i < hidden; ++i) {
        y[i] = tensor_value(emb, base + static_cast<size_t>(i));
    }
}

} // namespace cpu
} // namespace llm_inference
