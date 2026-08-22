//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoERouter.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

MoERouter::MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {
}

MoERoute MoERouter::forward(DeepseekSession &session, const GPUTensor &g_normed_f32) {
    const int input_size = static_cast<int>(g_normed_f32.rows());
    auto &s = session.scratch;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    GPUTensor g_router_logits_f32 = GPUTensor(
        s, scratch_key::kRouterLogits, {static_cast<int64_t>(input_size), static_cast<int64_t>(n_exp)}, DType::F32);
    // lw_.s_ffn_gate_inp->to_gpu(true);
    TensorTool::gemm(*lw_.s_ffn_gate_inp, g_normed_f32, g_router_logits_f32, s, scratch_key::kFfnInLowp,
                     "ds.gemm.router");

    const std::vector<int64_t> route_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(k)};
    GPUTensor g_top_idx_i32 = GPUTensor(s, scratch_key::kTopIdx, route_shape, DType::I32);
    GPUTensor g_top_w_f32 = GPUTensor(s, scratch_key::kTopW, route_shape, DType::F32);
    TensorTool::moe_router_topk(g_router_logits_f32, g_top_idx_i32, g_top_w_f32, n_exp, k, config_.routed_scaling);

    MoERoute route;
    route.expert_ids.resize(static_cast<size_t>(input_size) * k);
    route.weights.resize(static_cast<size_t>(input_size) * k);
    g_top_idx_i32.to_host(route.expert_ids.data(), "ds.moe.idx");
    g_top_w_f32.to_host(route.weights.data(), "ds.moe.w");
    return route;
}
