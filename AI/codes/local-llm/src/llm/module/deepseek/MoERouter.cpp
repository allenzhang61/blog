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

    auto g_router_logits_f32 = GPUTensor(s, scratch_key::kRouterLogits, {input_size, n_exp}, DType::F32);
    // lw_.s_ffn_gate_inp->to_gpu(true);
    TensorTool::gemm(*lw_.s_ffn_gate_inp, g_normed_f32, g_router_logits_f32, s, scratch_key::kFfnInLowp,
                     "ds.gemm.router");

    const std::vector<int64_t> route_shape = {input_size, k};
    auto g_top_idx_i32 = GPUTensor(s, scratch_key::kTopIdx, route_shape, DType::I32);
    auto g_top_w_f32 = GPUTensor(s, scratch_key::kTopW, route_shape, DType::F32);
    TensorTool::moe_router_topk(g_router_logits_f32, g_top_idx_i32, g_top_w_f32,
                                static_cast<int>(n_exp), static_cast<int>(k), config_.routed_scaling);

    MoERoute route;
    route.c_expert_ids_i32 = g_top_idx_i32.to_host(session.cpu_scratch, cpu_scratch_key::kMoeExpertIds, "ds.moe.idx");
    if (input_size == 1) {
        // decode：top_w 不回读 host，保留 device 指针供加权累加 kernel 直接读取；
        // 只回读了 k 个 int 索引（上面那次，host 侧按专家反量化权重切片需要），
        // 省掉每层一次 top_w 的 device 同步回读。
        route.d_weights_f32 = g_top_w_f32.data<float>();
        route.decode_device = true;
    } else {
        route.c_weights_f32 = g_top_w_f32.to_host(session.cpu_scratch, cpu_scratch_key::kMoeWeights, "ds.moe.w");
    }
    return route;
}
