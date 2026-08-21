//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoE.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/TensorTool.h"

#include <cstddef>

#include <cuda_runtime.h>

MoE::MoE(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config),
      weights_(weights),
      router_(weights, config),
      routed_experts_(weights, config),
      shared_experts_(weights, config) {}

void MoE::forward(DeepseekSession &session, const GPUTensor &hidden) {
    const int input_size = static_cast<int>(hidden.rows());
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};

    GPUTensor normed = GPUTensor::gpu_scratch(s, scratch_key::kNormed, act_shape);
    RMSNorm::forward(*weights_.ffn_norm, hidden, normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    MoERoute route = router_.forward(session, normed);

    GPUTensor moe_out = GPUTensor::gpu_scratch(s, scratch_key::kMoeOut, act_shape);
    float *d_moe_out = moe_out.gpu_f32();
    check_cuda(cudaMemset(d_moe_out, 0, static_cast<size_t>(input_size) * hidden_size * sizeof(float)), "ds.moe.zero");

    routed_experts_.forward(session, normed, route, moe_out);
    shared_experts_.forward(session, normed, moe_out);

    TensorTool::add(hidden, moe_out, hidden);
}
