//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_EMBEDDING_H
#define LOCAL_LLM_EMBEDDING_H

#include <vector>

#include "llm/qwen/QwenWeights.h"
#include "Module.h"

class CudaWeightPool;
class QwenForwardScratch;

// 词嵌入查表：token id -> hidden 向量。
// 权重 embed_tokens 形状 [vocab_size, hidden_size]。
// 因 tie_word_embeddings=true，同一份权重也被 LMHead 复用（见 LMHead）。
class Embedding : public Module {
public:
    Embedding(const WeightData &weight, CudaWeightPool *pool);

    // 按 token id 逐行拷贝嵌入到 d_out（device），形状 [tokens, hidden_size]。
    // h_token_ids 为 host 端 token id；内部负责搬运到 device 查表。
    void forward(const std::vector<int> &inputs, float *d_out, int hidden_size,
                 QwenForwardScratch &scratch);

private:
    const WeightData &weight_;
    CudaWeightPool *pool_ = nullptr;
};


#endif //LOCAL_LLM_EMBEDDING_H
