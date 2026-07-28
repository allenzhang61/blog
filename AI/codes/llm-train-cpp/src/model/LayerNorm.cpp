#include "llm/model/LayerNorm.hpp"

#include "llm/ops.hpp"

namespace llm {

LayerNorm::LayerNorm(int64_t emb_dim, Device device)
    : scale(Tensor::ones({emb_dim}, device, true)),
      shift(Tensor::zeros({emb_dim}, device, true)) {
}

// 对末维(emb_dim)做归一化，形状不变：x: (..., emb_dim) -> 返回: (..., emb_dim)
// scale/shift: (emb_dim)，逐元素缩放和平移
Tensor LayerNorm::forward(const Tensor& x) {
    return ops::layernorm(x, scale, shift, eps);
}

std::vector<Tensor*> LayerNorm::parameters() {
    return {&scale, &shift};
}

} // namespace llm
