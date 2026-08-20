//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <cstdint>

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/SessionBase.h"
#include "utils/sampling/Sampler.h"

namespace common {

LMHead::LMHead(const Tensor &weight)
    : weight_(weight) {}

int LMHead::forward(SessionBase &session, const Tensor &hidden, Sampler &sampler) {
    CudaScratch &scratch = session.scratch;

    const int vocab_size = static_cast<int>(weight_.shape[0]);

    Tensor logits = Tensor::gpu_scratch(scratch, scratch_key::kLogits, {static_cast<int64_t>(vocab_size)});
    weight_.to_gpu();
    weight_.gemm(hidden, logits, scratch, scratch_key::kLogitsInLowp, "lm_head");

    // logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    scratch.h_logits.resize(static_cast<size_t>(vocab_size));
    logits.to_host(scratch.h_logits.data(), "cudaMemcpy lm_head logits d2h 失败");
    return sampler.sample(scratch.h_logits.data(), vocab_size, session.outputs);
}

} // namespace common
