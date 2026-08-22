//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <cstdint>

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/SessionBase.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"
#include "utils/sampling/Sampler.h"

namespace common {

LMHead::LMHead(const StorageTensor &s_weight)
    : s_weight_(s_weight) {}

int LMHead::forward(SessionBase &session, const GPUTensor &g_hidden, Sampler &sampler) {
    CudaScratch &scratch = session.scratch;

    const int vocab_size = static_cast<int>(s_weight_.shape[0]);

    GPUTensor g_logits = GPUTensor(
        scratch, scratch_key::kLogits, {static_cast<int64_t>(vocab_size)}, DType::F32);
    TensorTool::gemm(s_weight_, g_hidden, g_logits, scratch, scratch_key::kLogitsInLowp, "lm_head");

    // g_logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    scratch.h_logits.resize(static_cast<size_t>(vocab_size));
    g_logits.to_host(scratch.h_logits.data(), "cudaMemcpy lm_head g_logits d2h 失败");
    return sampler.sample(scratch.h_logits.data(), vocab_size, session.output);
}

} // namespace common
