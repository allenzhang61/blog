//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenForwardScratch.h"
#include "utils/sampling/Sampler.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

LMHead::LMHead(const MFTensorView &weight, CudaWeightPool *pool)
    : weight_(weight), pool_(pool) {}

int LMHead::forward(const float *d_hidden, int hidden_size, QwenForwardScratch &scratch,
                    Sampler &sampler, const std::vector<int> &prev_tokens) {
    CudaWeight w = pool_->cached_weight(weight_)->try_dequant();
    // 复用 embed_tokens 权重 [vocab, hidden] 作为输出投影。
    const int vocab_size = static_cast<int>(weight_.shape[0]);

    // 输入激活转成权重 dtype（BF16/F16）后再投影（cublasGemmEx 要求同 dtype）；F32 权重直接透传。
    uint16_t *d_hidden_lowp = scratch.input_lowp_buffer.ensure(hidden_size, "lm_head in lowp");
    GemmInput in = prepare_gemm_input(d_hidden, d_hidden_lowp, hidden_size, w.type, nullptr);

    float *d_logits = scratch.y_buffer.ensure(static_cast<size_t>(vocab_size), "lm_head logits");
    gemm_weight(pool_->handle, w, in.ptr, d_logits, vocab_size, hidden_size, /*tokens=*/1, in.type, "lm_head");

    // logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    scratch.h_logits.resize(static_cast<size_t>(vocab_size));
    check_cuda(cudaMemcpy(scratch.h_logits.data(), d_logits, static_cast<size_t>(vocab_size) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy lm_head logits d2h 失败");
    return sampler.sample(scratch.h_logits.data(), vocab_size, prev_tokens);
}
