//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoERouter.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

MoERouter::MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

MoERoute MoERouter::forward(DeepseekSession &session, const GPUTensor &normed) {
    const int input_size = static_cast<int>(normed.rows());
    auto &s = session.scratch;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    GPUTensor router_logits = GPUTensor::gpu_scratch(
        s, scratch_key::kRouterLogits, {static_cast<int64_t>(input_size), static_cast<int64_t>(n_exp)});
    lw_.ffn_gate_inp->to_gpu();
    TensorTool::gemm(*lw_.ffn_gate_inp, normed, router_logits, s, scratch_key::kFfnInLowp, "ds.gemm.router");

    const std::vector<int64_t> route_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(k)};
    GPUTensor top_idx = GPUTensor::gpu_scratch(s, scratch_key::kTopIdx, route_shape, DType::I32);
    GPUTensor top_w = GPUTensor::gpu_scratch(s, scratch_key::kTopW, route_shape);
    TensorTool::moe_router_topk(router_logits, top_idx, top_w, n_exp, k, config_.routed_scaling);

    MoERoute route;
    route.expert_ids.resize(static_cast<size_t>(input_size) * k);
    route.weights.resize(static_cast<size_t>(input_size) * k);
    top_idx.to_host(route.expert_ids.data(), "ds.moe.idx");
    top_w.to_host(route.weights.data(), "ds.moe.w");
    return route;
}
