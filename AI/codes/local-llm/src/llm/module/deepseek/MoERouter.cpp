//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoERouter.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"

#include <cstddef>

#include <cuda_runtime.h>

MoERouter::MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config), lw_(weights), pool_(pool) {}

MoERoute MoERouter::forward(DeepseekSession &session, const float *d_normed, int input_size) {
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    float *d_router_logits = s.ensure<float>(scratch_key::kRouterLogits, static_cast<size_t>(input_size) * n_exp);
    {
        CudaWeight ffn_gate_inp = pool_->cached_weight(*lw_.ffn_gate_inp)->try_dequant();
        uint16_t *d_ffn_in_lowp = s.ensure<uint16_t>(scratch_key::kFfnInLowp, static_cast<size_t>(input_size) * hidden_size);
        GemmInput router_in = prepare_gemm_input(d_normed, d_ffn_in_lowp, input_size * hidden_size, ffn_gate_inp.type, nullptr);
        gemm_weight(pool_->handle, ffn_gate_inp, router_in.ptr, d_router_logits, n_exp, hidden_size, input_size, router_in.type, "ds.gemm.router");
    }

    int *d_top_idx = s.ensure<int>(scratch_key::kTopIdx, static_cast<size_t>(input_size) * k);
    float *d_top_w = s.ensure<float>(scratch_key::kTopW, static_cast<size_t>(input_size) * k);
    launch_moe_router_topk(d_router_logits, d_top_idx, d_top_w, input_size, n_exp, k, config_.routed_scaling, nullptr);

    MoERoute route;
    route.expert_ids.resize(static_cast<size_t>(input_size) * k);
    route.weights.resize(static_cast<size_t>(input_size) * k);
    cuda_memcpy_d2h(route.expert_ids.data(), d_top_idx, route.expert_ids.size() * sizeof(int),
                    "ds.moe.idx");
    cuda_memcpy_d2h(route.weights.data(), d_top_w, route.weights.size() * sizeof(float),
                    "ds.moe.w");
    return route;
}
