//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoE.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekTrace.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

MoE::MoE(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config),
      weights_(weights),
      router_(weights, config),
      routed_experts_(weights, config),
      shared_experts_(weights, config) {}

void MoE::forward(DeepseekSession &session, const GPUTensor &g_hidden_f32) {
    const int64_t input_size = g_hidden_f32.rows();
    auto &s = session.cuda_scratch;
    const int64_t hidden_size = config_.hidden_size;

    const auto g_normed_f32 = GPUTensor(s, scratch_key::kNormed, {input_size, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_ffn_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    deepseek_trace::tensor(session, g_normed_f32, "moe_normed", session.trace_pos, session.trace_layer);

    const MoERoute route = router_.forward(session, g_normed_f32);

    const auto g_moe_out_f32 = GPUTensor(s, scratch_key::kMoeOut, {input_size, hidden_size}, DType::F32);
    float *d_moe_out = g_moe_out_f32.data<float>();
    check_cuda(cudaMemsetAsync(d_moe_out, 0, static_cast<size_t>(input_size) * hidden_size * sizeof(float),
                               get_current_cuda_stream()),
               "ds.moe.zero");

    routed_experts_.forward(session, g_normed_f32, route, g_moe_out_f32);
    shared_experts_.forward(session, g_normed_f32, g_moe_out_f32);
    deepseek_trace::tensor(session, g_moe_out_f32, "moe_out", session.trace_pos, session.trace_layer);

    TensorTool::add(g_hidden_f32, g_moe_out_f32, g_hidden_f32);
    deepseek_trace::tensor(session, g_hidden_f32, "moe_post_add", session.trace_pos, session.trace_layer);
}
