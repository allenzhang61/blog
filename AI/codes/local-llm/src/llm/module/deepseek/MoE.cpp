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

MoE::MoE(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool)
    : config_(config),
      weights_(weights),
      pool_(pool),
      router_(config, pool),
      routed_experts_(config, pool),
      shared_experts_(config, pool) {}

void MoE::forward(DeepseekSession &session, int layer, float *d_hidden, int input_size) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int hidden_size = config_.hidden_size;

    float *d_normed = s.ensure<float>(scratch_key::kNormed, static_cast<size_t>(input_size) * hidden_size, "ds.normed");
    RMSNorm::forward(pool_, *lw.ffn_norm, d_hidden, d_normed, input_size, hidden_size,
                     config_.rms_norm_eps, /*one_plus=*/false);

    MoERoute route = router_.forward(session, lw, d_normed, input_size);

    float *d_moe = s.ensure<float>(scratch_key::kMoeOut, static_cast<size_t>(input_size) * hidden_size, "ds.moe_out");
    check_cuda(cudaMemset(d_moe, 0, static_cast<size_t>(input_size) * hidden_size * sizeof(float)), "ds.moe.zero");

    routed_experts_.forward(session, lw, d_normed, route, input_size, d_moe);
    shared_experts_.forward(session, lw, d_normed, input_size, d_moe);

    launch_add(d_hidden, d_moe, d_hidden, input_size * hidden_size, nullptr);
}
