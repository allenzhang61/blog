//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <cstdint>

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/mem/SessionBase.h"
#include "backend/cuda/ops/gemm.h"
#include "utils/sampling/Sampler.h"

namespace common {

LMHead::LMHead(const Tensor &weight)
    : weight_(weight) {}

int LMHead::forward(SessionBase &session, const Tensor &hidden, Sampler &sampler) {
    CudaScratch &scratch = session.scratch;
    CudaWeight w = weight_.cached_weight()->try_dequant();

    const int hidden_size = static_cast<int>(hidden.cols());
    const int vocab_size = static_cast<int>(weight_.shape[0]);

    // 输入激活转成权重 dtype（BF16/F16）后再投影（cublasGemmEx 要求同 dtype）；F32 权重直接透传。
    uint16_t *d_hidden_lowp =
        scratch.ensure<uint16_t>(scratch_key::kLogitsInLowp, hidden_size);
    GemmInput in = prepare_gemm_input(hidden.gpu_f32(), d_hidden_lowp, hidden_size, w.type, nullptr);

    float *d_logits = scratch.ensure<float>(scratch_key::kLogits, static_cast<size_t>(vocab_size));
    gemm_weight(global_cuda_weight_pool().handle, w, in.ptr, d_logits, vocab_size, hidden_size, /*tokens=*/1, in.type, "lm_head");

    // logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    scratch.h_logits.resize(static_cast<size_t>(vocab_size));
    cuda_memcpy_d2h(scratch.h_logits.data(), d_logits, static_cast<size_t>(vocab_size) * sizeof(float),
                    "cudaMemcpy lm_head logits d2h 失败");
    return sampler.sample(scratch.h_logits.data(), vocab_size, session.outputs);
}

} // namespace common
