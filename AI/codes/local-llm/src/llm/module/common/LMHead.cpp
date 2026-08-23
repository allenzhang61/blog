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

int LMHead::forward(const StorageTensor &s_weight, SessionBase &session,
                    const GPUTensor &g_hidden_f32, Sampler &sampler) {
    CudaScratch &scratch = session.scratch;

    const int vocab_size = static_cast<int>(s_weight.shape[0]);

    GPUTensor g_logits_f32 = GPUTensor(
        scratch, scratch_key::kLogits, {static_cast<int64_t>(vocab_size)}, DType::F32);
    TensorTool::gemm(s_weight, g_hidden_f32, g_logits_f32, scratch, scratch_key::kLogitsInLowp, "lm_head");

    // g_logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    scratch.h_logits.resize(static_cast<size_t>(vocab_size));
    g_logits_f32.to_host(scratch.h_logits.data(), "cudaMemcpy lm_head g_logits d2h 失败");
    return sampler.sample(scratch.h_logits.data(), vocab_size, session.output);
}

void LMHead::forward_argmax_device(const StorageTensor &s_weight, SessionBase &session,
                                   const GPUTensor &g_hidden_f32, int *d_out_token, void *stream) {
    CudaScratch &scratch = session.scratch;
    const int vocab_size = static_cast<int>(s_weight.shape[0]);

    GPUTensor g_logits_f32 = GPUTensor(
        scratch, scratch_key::kLogits, {static_cast<int64_t>(vocab_size)}, DType::F32);
    TensorTool::gemm(s_weight, g_hidden_f32, g_logits_f32, scratch, scratch_key::kLogitsInLowp, "lm_head");
    // GPU argmax 直接把下一个 token id 写到 device buffer，全程留 device，供 CUDA Graph replay。
    TensorTool::argmax(g_logits_f32, d_out_token, vocab_size, stream);
}

} // namespace common
