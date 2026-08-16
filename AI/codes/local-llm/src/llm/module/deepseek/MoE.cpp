//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoE.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/deepseek/RMSNorm.h"

#include <cstddef>

#include <cuda_runtime.h>

MoE::MoE(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
         deepseek::RMSNorm *rms_norm)
    : config_(config),
      weights_(weights),
      pool_(pool),
      rms_norm_(rms_norm),
      router_(config, pool),
      routed_experts_(config, pool),
      shared_experts_(config, pool) {}

void MoE::forward(DeepseekSession &session, int layer, float *d_hidden, int tokens) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int H = config_.hidden_size;

    float *d_normed = s.normed.ensure(static_cast<size_t>(tokens) * H, "ds.normed");
    rms_norm_->forward(*lw.ffn_norm, d_hidden, d_normed, tokens, H);

    MoERoute route = router_.forward(session, lw, d_normed, tokens);

    float *d_moe = s.moe_out.ensure(static_cast<size_t>(tokens) * H, "ds.moe_out");
    check_cuda(cudaMemset(d_moe, 0, static_cast<size_t>(tokens) * H * sizeof(float)), "ds.moe.zero");

    routed_experts_.forward(session, lw, d_normed, route, tokens, d_moe);
    shared_experts_.forward(session, lw, d_normed, tokens, d_moe);

    launch_add(d_hidden, d_moe, d_hidden, tokens * H, nullptr);
}
