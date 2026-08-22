//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

namespace common {

Embedding::Embedding(const StorageTensor &s_weight)
    : s_weight_(s_weight) {}

void Embedding::forward(CPUTensor c_input, const GPUTensor &g_hidden, CudaScratch &scratch) {
    TensorTool::embedding_lookup(s_weight_, c_input, g_hidden, scratch);
}

} // namespace common
