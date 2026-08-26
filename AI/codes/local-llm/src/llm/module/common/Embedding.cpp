//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

void Embedding::forward(const StorageTensor &s_weight, CPUTensor c_input_i32,
                        const GPUTensor &g_hidden_f32, CudaScratch &scratch) {
    TensorTool::embedding_lookup(s_weight, c_input_i32, g_hidden_f32, scratch);
}
