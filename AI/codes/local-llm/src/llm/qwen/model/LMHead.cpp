//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

LMHead::LMHead(const WeightData &weight, CudaWeightPool *pool)
    : weight_(weight), pool_(pool) {}

int LMHead::forward(const float *d_hidden, int hidden_size, QwenForwardScratch &scratch) {
    CudaWeight *w = pool_->cached_weight(weight_);
    if (!w) {
        throw std::runtime_error("LMHead 权重上传失败：" + weight_.info->name);
    }
    // 复用 embed_tokens 权重 [vocab, hidden] 作为输出投影。
    const int vocab = static_cast<int>(weight_.info->shape[0]);

    // 输入激活转成权重 dtype（BF16/F16）后再投影（cublasGemmEx 要求同 dtype）。
    uint16_t *d_in_lowp = scratch.input_lowp_buffer.ensure(hidden_size, "lm_head in lowp");
    to_weight_lowp(d_hidden, d_in_lowp, hidden_size, *w, nullptr);

    float *d_logits = scratch.y_buffer.ensure(static_cast<size_t>(vocab), "lm_head logits");
    gemm_weight(pool_->handle, *w, vocab, hidden_size, d_in_lowp, w->type, /*tokens=*/1, d_logits);

    // argmax：分块归约。block_values/indices 上限 1024（与 kernel 内一致）。
    float *d_block_val = scratch.argmax_block_values.ensure(1024, "argmax block values");
    int *d_block_idx = scratch.argmax_block_indices.ensure(1024, "argmax block indices");
    float *d_best_val = scratch.argmax_best_value.ensure(1, "argmax best value");
    int *d_best_idx = scratch.argmax_best_index.ensure(1, "argmax best index");

    launch_argmax(d_logits, vocab, d_block_val, d_block_idx, d_best_val, d_best_idx,
                  /*stream=*/nullptr);

    int h_token_id = 0;
    check_cuda(cudaMemcpy(&h_token_id, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy argmax token id 失败");
    return h_token_id;
}
