//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/mem/CudaScratch.h"

namespace common {

Embedding::Embedding(const Tensor &weight)
    : weight_(weight) {}

void Embedding::forward(Tensor input, const Tensor &hidden, CudaScratch &scratch) {
    input.to_gpu(scratch, scratch_key::kInput, "cudaMemcpy embedding token ids 失败");
    weight_.to_gpu();
    weight_.embedding_lookup(input, hidden);
}

} // namespace common
