//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

void Embedding::forward(const StorageTensor &s_weight, CPUTensor c_input_i32,
                        const GPUTensor &g_hidden_f32, CudaScratch &scratch) {
    GPUTensor g_input_i32 = c_input_i32.to_gpu(scratch, scratch_key::kInput,
                                               "cudaMemcpy embedding token ids 失败");
    TensorTool::embedding_lookup(s_weight, g_input_i32, g_hidden_f32);
}
