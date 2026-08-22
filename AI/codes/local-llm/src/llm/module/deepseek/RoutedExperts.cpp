//
// Created by zhangyoulun on 15/8/2026.
//

#include "RoutedExperts.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "format/MF.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

StorageTensor s_expert_tensor_view(const StorageTensor &s_weight, int expert, int n_experts) {
    const int64_t n_per_expert = s_weight.numel() / n_experts;
    const size_t bytes_per_expert = s_weight.nbytes / static_cast<size_t>(n_experts);

    return s_weight.slice(static_cast<size_t>(expert) * bytes_per_expert,
                        {n_per_expert},
                        bytes_per_expert,
                        s_weight.name + ".e" + std::to_string(expert));
}

} // namespace

RoutedExperts::RoutedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

void RoutedExperts::forward(DeepseekSession &session, const GPUTensor &g_normed, const MoERoute &route,
                            const GPUTensor &g_moe) {
    const int input_size = static_cast<int>(g_normed.rows());
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.expert_ffn;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    GPUTensor g_gate_out = GPUTensor(s, scratch_key::kGate, {1, static_cast<int64_t>(ffn)}, DType::F32);
    GPUTensor g_up_out = GPUTensor(s, scratch_key::kUp, {1, static_cast<int64_t>(ffn)}, DType::F32);
    GPUTensor g_act = GPUTensor(s, scratch_key::kAct, {1, static_cast<int64_t>(ffn)}, DType::F32);
    GPUTensor g_expert_out = GPUTensor(
        s, scratch_key::kExpertOut, {1, static_cast<int64_t>(hidden_size)}, DType::F32);

    for (int tok = 0; tok < input_size; ++tok) {
        const size_t token_offset = static_cast<size_t>(tok) * hidden_size * sizeof(float);
        GPUTensor g_tok_in = GPUTensor(g_normed, token_offset, {1, static_cast<int64_t>(hidden_size)});
        for (int r = 0; r < k; ++r) {
            const size_t route_idx = static_cast<size_t>(tok) * k + r;
            const int e = route.expert_ids[route_idx];
            const float w = route.weights[route_idx];
            StorageTensor s_gate = s_expert_tensor_view(*lw_.s_ffn_gate_exps, e, n_exp);
            s_gate.shape = {static_cast<int64_t>(ffn), static_cast<int64_t>(hidden_size)};
            TensorTool::gemm(s_gate, g_tok_in, g_gate_out, s, scratch_key::kFfnInLowp, "ds.gemm.egate");
            StorageTensor s_up = s_expert_tensor_view(*lw_.s_ffn_up_exps, e, n_exp);
            s_up.shape = {static_cast<int64_t>(ffn), static_cast<int64_t>(hidden_size)};
            TensorTool::gemm(s_up, g_tok_in, g_up_out, s, scratch_key::kFfnInLowp, "ds.gemm.eup");
            TensorTool::silu_mul(g_gate_out, g_up_out, g_act);
            StorageTensor s_down = s_expert_tensor_view(*lw_.s_ffn_down_exps, e, n_exp);
            s_down.shape = {static_cast<int64_t>(hidden_size), static_cast<int64_t>(ffn)};
            TensorTool::gemm(s_down, g_act, g_expert_out, s, scratch_key::kActLowp, "ds.gemm.edown");
            GPUTensor g_tok_moe = GPUTensor(g_moe, token_offset, {1, static_cast<int64_t>(hidden_size)});
            TensorTool::moe_accumulate(g_expert_out, w, g_tok_moe);
        }
    }
}
