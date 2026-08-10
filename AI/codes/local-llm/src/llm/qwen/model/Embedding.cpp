//
// Created by zhangyoulun on 9/8/2026.
//

#include "Embedding.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

Embedding::Embedding(const WeightData &weight, CudaWeightPool *pool)
    : weight_(weight), pool_(pool) {}

void Embedding::forward(const std::vector<int> &h_token_ids, float *d_out, int hidden_size) {
    CudaWeight *table = pool_->cached_weight(weight_);
    if (!table) {
        throw std::runtime_error("Embedding 权重上传失败：" + weight_.info->name);
    }
    // 权重形状 [vocab, hidden]，dtype bf16/f16。
    const int vocab = static_cast<int>(weight_.info->shape[0]);
    const int lowp_type = (weight_.info->dtype == DType::F16) ? 1 : 0;
    const int tokens = static_cast<int>(h_token_ids.size());

    // token id 搬到 device（一次前向的小临时缓冲，用完即释放）。
    int *d_ids = nullptr;
    const size_t ids_bytes = static_cast<size_t>(tokens) * sizeof(int);
    check_cuda(cudaMalloc(&d_ids, ids_bytes), "cudaMalloc embedding token ids 失败");
    check_cuda(cudaMemcpy(d_ids, h_token_ids.data(), ids_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy embedding token ids 失败");

    launch_embedding_lookup(static_cast<const uint16_t *>(table->ptr), d_ids, d_out,
                            tokens, vocab, hidden_size, lowp_type, /*stream=*/nullptr);

    check_cuda(cudaDeviceSynchronize(), "embedding_lookup 同步失败");
    cudaFree(d_ids);
}
