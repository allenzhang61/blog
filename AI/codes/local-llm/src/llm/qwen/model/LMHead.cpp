//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "llm/sampling/Sampler.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

LMHead::LMHead(const WeightData &weight, CudaWeightPool *pool)
    : weight_(weight), pool_(pool) {}

int LMHead::forward(const float *d_hidden, int hidden_size, QwenForwardScratch &scratch,
                    Sampler &sampler, const std::vector<int> &prev_tokens) {
    CudaWeight *w = pool_->cached_weight(weight_);
    if (!w) {
        throw std::runtime_error("LMHead 权重上传失败：" + weight_.name);
    }
    // 复用 embed_tokens 权重 [vocab, hidden] 作为输出投影。
    const int vocab = static_cast<int>(weight_.shape[0]);

    // 输入激活转成权重 dtype（BF16/F16）后再投影（cublasGemmEx 要求同 dtype）。
    uint16_t *d_in_lowp = scratch.input_lowp_buffer.ensure(hidden_size, "lm_head in lowp");
    to_weight_lowp(d_hidden, d_in_lowp, hidden_size, *w, nullptr);

    float *d_logits = scratch.y_buffer.ensure(static_cast<size_t>(vocab), "lm_head logits");
    gemm_weight(pool_->handle, *w, vocab, hidden_size, d_in_lowp, w->type, /*tokens=*/1, d_logits, "lm_head");

    // logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    scratch.h_logits.resize(static_cast<size_t>(vocab));
    check_cuda(cudaMemcpy(scratch.h_logits.data(), d_logits, static_cast<size_t>(vocab) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy lm_head logits d2h 失败");
    return sampler.sample(scratch.h_logits.data(), vocab, prev_tokens);
}
