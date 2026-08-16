//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoERouter.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"

#include <cstddef>

#include <cuda_runtime.h>

MoERouter::MoERouter(const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config), pool_(pool) {}

MoERoute MoERouter::forward(DeepseekSession &session, const DeepseekLayerWeights &weights,
                            const float *d_normed, int input_size) {
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    float *d_router = s.router_logits.ensure(static_cast<size_t>(input_size) * n_exp, "ds.router");
    {
        CudaWeight w = pool_->cached_weight(*weights.ffn_gate_inp)->try_dequant();
        uint16_t *xlow = s.ffn_in_lowp.ensure(static_cast<size_t>(input_size) * hidden_size, "ds.ffn_in_lowp");
        GemmInput router_in = prepare_gemm_input(d_normed, xlow, input_size * hidden_size, w.type, nullptr);
        gemm_weight(pool_->handle, w, router_in.ptr, d_router, n_exp, hidden_size, input_size, router_in.type, "ds.gemm.router");
    }

    int *d_topidx = s.top_idx.ensure(static_cast<size_t>(input_size) * k, "ds.topidx");
    float *d_topw = s.top_w.ensure(static_cast<size_t>(input_size) * k, "ds.topw");
    launch_moe_router_topk(d_router, d_topidx, d_topw, input_size, n_exp, k, config_.routed_scaling, nullptr);

    MoERoute route;
    route.expert_ids.resize(static_cast<size_t>(input_size) * k);
    route.weights.resize(static_cast<size_t>(input_size) * k);
    check_cuda(cudaMemcpy(route.expert_ids.data(), d_topidx, route.expert_ids.size() * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "ds.moe.idx");
    check_cuda(cudaMemcpy(route.weights.data(), d_topw, route.weights.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "ds.moe.w");
    return route;
}
