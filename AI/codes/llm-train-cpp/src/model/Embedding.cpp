#include "llm/model/Embedding.hpp"

#include "llm/ops.hpp"

namespace llm {

// 权重初始化对齐 PyTorch nn.Embedding 的默认：weight ~ N(0, 1)（标准正态）。
// 原先的 std=0.02 会让 embedding 信号过弱，在深层网络中被 LayerNorm 归一化后失去区分度。
Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device)
    : weight(Tensor::randn({num_embeddings, embedding_dim}, 1.0, device, true)) {
}

// ids: (...) 整数索引 -> 返回: (..., embedding_dim)
// weight: (num_embeddings, embedding_dim)，按 id 查表取对应行
Tensor Embedding::forward(const Tensor& ids) {
    return ops::embedding(ids, weight);
}

std::vector<Tensor*> Embedding::parameters() {
    return {&weight};
}

} // namespace llm
