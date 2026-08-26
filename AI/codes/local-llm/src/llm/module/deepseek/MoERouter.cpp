//
// Created by zhangyoulun on 15/8/2026.
//

#include "MoERouter.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/CPUScratch.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

MoERouter::MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {
}

MoERoute MoERouter::forward(DeepseekSession &session, const GPUTensor &g_normed_f32) {
    const int64_t input_size = g_normed_f32.rows();
    auto &s = session.cuda_scratch;
    const int64_t n_exp = config_.expert_count;
    const int64_t k = config_.expert_used;

    GPUTensor g_router_logits_f32 = GPUTensor(
        s, scratch_key::kRouterLogits, {input_size, n_exp}, DType::F32);
    // lw_.s_ffn_gate_inp->to_gpu(true);
    TensorTool::gemm(*lw_.s_ffn_gate_inp, g_normed_f32, g_router_logits_f32, s, scratch_key::kFfnInLowp,
                     "ds.gemm.router");

    const std::vector<int64_t> route_shape = {input_size, k};
    GPUTensor g_top_idx_i32 = GPUTensor(s, scratch_key::kTopIdx, route_shape, DType::I32);
    GPUTensor g_top_w_f32 = GPUTensor(s, scratch_key::kTopW, route_shape, DType::F32);
    TensorTool::moe_router_topk(g_router_logits_f32, g_top_idx_i32, g_top_w_f32,
                                static_cast<int>(n_exp), static_cast<int>(k), config_.routed_scaling);

    MoERoute route;
    route.c_expert_ids_i32 = g_top_idx_i32.to_host(session.cpu_scratch, cpu_scratch_key::kMoeExpertIds, "ds.moe.idx");
    route.c_weights_f32 = g_top_w_f32.to_host(session.cpu_scratch, cpu_scratch_key::kMoeWeights, "ds.moe.w");
    return route;
}
