//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoERouter.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

MoERouter::MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

MoERoute MoERouter::forward(DeepseekSession &session, const Tensor &normed) {
    const int input_size = static_cast<int>(normed.rows());
    auto &s = session.scratch;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    Tensor router_logits = Tensor::gpu_scratch(
        s, scratch_key::kRouterLogits, {static_cast<int64_t>(input_size), static_cast<int64_t>(n_exp)});
    lw_.ffn_gate_inp->to_gpu();
    lw_.ffn_gate_inp->gemm(normed, router_logits, s, scratch_key::kFfnInLowp, "ds.gemm.router");

    const std::vector<int64_t> route_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(k)};
    Tensor top_idx = Tensor::gpu_scratch(s, scratch_key::kTopIdx, route_shape, DType::I32);
    Tensor top_w = Tensor::gpu_scratch(s, scratch_key::kTopW, route_shape);
    launch_moe_router_topk(router_logits.gpu_f32(), top_idx.gpu_i32(), top_w.gpu_f32(),
                           input_size, n_exp, k, config_.routed_scaling, nullptr);

    MoERoute route;
    route.expert_ids.resize(static_cast<size_t>(input_size) * k);
    route.weights.resize(static_cast<size_t>(input_size) * k);
    top_idx.to_host(route.expert_ids.data(), "ds.moe.idx");
    top_w.to_host(route.weights.data(), "ds.moe.w");
    return route;
}
