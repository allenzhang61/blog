//
// Created by zhangyoulun on 9/8/2026.
//

#include "LMHead.h"

#include <cstdint>

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/SessionBase.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekTrace.h"
#include "tensor/CPUScratch.h"
#include "tensor/CPUTensor.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"
#include "utils/sampling/Sampler.h"

int LMHead::forward(const StorageTensor &s_weight, SessionBase &session,
                    const GPUTensor &g_hidden_f32, Sampler &sampler) {
    CudaScratch &scratch = session.cuda_scratch;

    const int vocab_size = static_cast<int>(s_weight.shape[0]);

    GPUTensor g_logits_f32 = GPUTensor(
        scratch, scratch_key::kLogits, {static_cast<int64_t>(vocab_size)}, DType::F32);
    TensorTool::gemm(s_weight, g_hidden_f32, g_logits_f32, scratch, scratch_key::kLogitsInLowp, "lm_head");
    if (auto *deepseek_session = dynamic_cast<DeepseekSession *>(&session)) {
        deepseek_trace::tensor(*deepseek_session, g_logits_f32, "logits",
                               deepseek_session->trace_pos, deepseek_session->trace_layer);
    }

    // g_logits 拷回 host，交由 Sampler 做温度/top-k/top-p/重复惩罚（greedy 时内部走 argmax）。
    CPUTensor h_logits_f32 = g_logits_f32.to_host(session.cpu_scratch, cpu_scratch_key::kLmHeadLogits,
                                                  "cudaMemcpy lm_head g_logits d2h 失败");
    float *logits = h_logits_f32.data<float>();
    return sampler.sample(logits, vocab_size, session.h_output_i32_);
}

void LMHead::forward_argmax_device(const StorageTensor &s_weight, SessionBase &session,
                                   const GPUTensor &g_hidden_f32, int *d_out_token, void *stream) {
    CudaScratch &scratch = session.cuda_scratch;
    const int vocab_size = static_cast<int>(s_weight.shape[0]);

    GPUTensor g_logits_f32 = GPUTensor(
        scratch, scratch_key::kLogits, {static_cast<int64_t>(vocab_size)}, DType::F32);
    TensorTool::gemm(s_weight, g_hidden_f32, g_logits_f32, scratch, scratch_key::kLogitsInLowp, "lm_head");
    if (auto *deepseek_session = dynamic_cast<DeepseekSession *>(&session)) {
        deepseek_trace::tensor(*deepseek_session, g_logits_f32, "logits",
                               deepseek_session->trace_pos, deepseek_session->trace_layer);
    }
    // GPU argmax 直接把下一个 token id 写到 device buffer，全程留 device，供 CUDA Graph replay。
    TensorTool::argmax(g_logits_f32, d_out_token, vocab_size, stream);
}
