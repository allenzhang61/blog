#include "llm/model/Embedding.hpp"

#include "llm/ops.hpp"

namespace llm {

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device)
    : weight(Tensor::randn({num_embeddings, embedding_dim}, 0.02, device, true)) {
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
