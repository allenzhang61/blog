//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

namespace common {

Embedding::Embedding(const DiskTensor &weight)
    : weight_(weight) {}

void Embedding::forward(CPUTensor input, const GPUTensor &hidden, CudaScratch &scratch) {
    TensorTool::embedding_lookup(weight_, input, hidden, scratch);
}

} // namespace common
