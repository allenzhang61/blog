//
// Created by zhangyoulun on 9/8/2026.
//

#include "Embedding.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

Embedding::Embedding(const WeightData &weight, CudaWeightPool *pool)
    : weight_(weight), pool_(pool) {}

void Embedding::forward(const std::vector<int> &inputs, float *d_out, int hidden_size,
                        QwenForwardScratch &scratch) {
    CudaWeight *table = pool_->cached_weight(weight_);
    if (!table) {
        throw std::runtime_error("Embedding 权重上传失败：" + weight_.name);
    }
    // 权重形状 [vocab, hidden]，dtype bf16/f16。
    const int vocab = static_cast<int>(weight_.shape[0]);
    const int lowp_type = (weight_.dtype == DType::F16) ? 1 : 0;
    const size_t input_size = inputs.size();

    // token id 搬到 device，使用 forward scratch 复用临时缓冲。
    int *d_inputs = scratch.inputs.ensure(input_size, "embedding token ids");
    check_cuda(cudaMemcpy(d_inputs, inputs.data(), input_size * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy embedding token ids 失败");

    launch_embedding_lookup(static_cast<const uint16_t *>(table->ptr), d_inputs, d_out,
                            static_cast<int>(input_size), vocab, hidden_size, lowp_type, /*stream=*/nullptr);

    check_cuda(cudaDeviceSynchronize(), "embedding_lookup 同步失败");
}
