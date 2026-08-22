//
// Created by zhangyoulun on 15/8/2026.
//

#include "SharedExperts.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <vector>

SharedExperts::SharedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

void SharedExperts::forward(DeepseekSession &session, const GPUTensor &g_normed_f32, const GPUTensor &g_moe_f32) {
    const int input_size = static_cast<int>(g_normed_f32.rows());
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int shared_ffn = config_.shared_ffn();

    const std::vector<int64_t> shared_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(shared_ffn)};
    GPUTensor g_gate_f32 = GPUTensor(s, scratch_key::kGate, shared_shape, DType::F32);
    GPUTensor g_up_f32 = GPUTensor(s, scratch_key::kUp, shared_shape, DType::F32);
    TensorTool::gemm(*lw_.s_ffn_gate_shexp, g_normed_f32, g_gate_f32, s, scratch_key::kFfnInLowp, "ds.gemm.sgate");
    TensorTool::gemm(*lw_.s_ffn_up_shexp, g_normed_f32, g_up_f32, s, scratch_key::kFfnInLowp, "ds.gemm.sup");

    GPUTensor g_act_f32 = GPUTensor(s, scratch_key::kAct, shared_shape, DType::F32);
    TensorTool::silu_mul(g_gate_f32, g_up_f32, g_act_f32);

    GPUTensor g_ffn_out_f32 = GPUTensor(
        s, scratch_key::kFfnOut, {static_cast<int64_t>(input_size), static_cast<int64_t>(hidden_size)}, DType::F32);
    TensorTool::gemm(*lw_.s_ffn_down_shexp, g_act_f32, g_ffn_out_f32, s, scratch_key::kActLowp, "ds.gemm.sdown");
    TensorTool::add(g_moe_f32, g_ffn_out_f32, g_moe_f32);
}
