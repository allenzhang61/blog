#include "llm/model/LayerNorm.hpp"

#include "llm/ops.hpp"

namespace llm {

LayerNorm::LayerNorm(int64_t emb_dim, Device device)
    : scale(Tensor::ones({emb_dim}, device, true)),
      shift(Tensor::zeros({emb_dim}, device, true)) {
}

Tensor LayerNorm::forward(const Tensor& x) {
    return ops::layernorm(x, scale, shift, eps);
}

std::vector<Tensor*> LayerNorm::parameters() {
    return {&scale, &shift};
}

} // namespace llm
