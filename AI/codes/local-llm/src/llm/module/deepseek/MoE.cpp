//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoE.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"

#include <cstddef>

#include <cuda_runtime.h>

MoE::MoE(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config),
      weights_(weights),
      pool_(pool),
      router_(weights, config, pool),
      routed_experts_(weights, config, pool),
      shared_experts_(weights, config, pool) {}

void MoE::forward(DeepseekSession &session, float *d_hidden, int input_size) {
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;

    float *d_normed = s.ensure<float>(scratch_key::kNormed, static_cast<size_t>(input_size) * hidden_size);
    RMSNorm::forward(pool_, *weights_.ffn_norm, d_hidden, d_normed, input_size, hidden_size,
                     config_.rms_norm_eps, /*one_plus=*/false);

    MoERoute route = router_.forward(session, d_normed, input_size);

    float *d_moe_out = s.ensure<float>(scratch_key::kMoeOut, static_cast<size_t>(input_size) * hidden_size);
    check_cuda(cudaMemset(d_moe_out, 0, static_cast<size_t>(input_size) * hidden_size * sizeof(float)), "ds.moe.zero");

    routed_experts_.forward(session, d_normed, route, input_size, d_moe_out);
    shared_experts_.forward(session, d_normed, input_size, d_moe_out);

    launch_add(d_hidden, d_moe_out, d_hidden, input_size * hidden_size, nullptr);
}
