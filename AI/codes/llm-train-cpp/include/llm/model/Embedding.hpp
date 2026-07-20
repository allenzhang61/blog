#pragma once

#include "llm/model/Module.hpp"

namespace llm {

// Embedding 层。
// 将 token id 映射为可训练的向量表示。
class Embedding : public Module {
public:
    // 查表矩阵，形状是 [num_embeddings, embedding_dim]。
    Tensor weight;

    // num_embeddings 是词表大小，embedding_dim 是每个 token 的向量维度。
    Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device = {});

    // 输入 ids 形状通常是 [batch, seq_len]，输出是 [batch, seq_len, embedding_dim]。
    Tensor forward(const Tensor& ids);

    // 返回 weight，供优化器更新。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
