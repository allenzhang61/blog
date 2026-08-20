//
// Created by zhangyoulun on 15/8/2026.
//

#include "RoutedExperts.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"
#include "format/MF.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

int64_t tensor_elements(const Tensor &tensor) {
    int64_t n = 1;
    for (int64_t dim : tensor.shape) {
        n *= dim;
    }
    return n;
}

Tensor expert_tensor_view(const Tensor &weight, int expert, int n_experts) {
    const int64_t n_per_expert = tensor_elements(weight) / n_experts;
    const size_t bytes_per_expert = weight.nbytes / static_cast<size_t>(n_experts);

    Tensor view = weight;
    view.name = weight.name + ".e" + std::to_string(expert);
    view.shape = {n_per_expert};
    view.disk_data = weight.disk_data + static_cast<size_t>(expert) * bytes_per_expert;
    view.nbytes = bytes_per_expert;
    return view;
}

} // namespace

RoutedExperts::RoutedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

void RoutedExperts::forward(DeepseekSession &session, const Tensor &normed, const MoERoute &route,
                            const Tensor &moe) {
    const float *d_normed = normed.gpu_f32();
    float *d_moe = moe.gpu_f32();
    const int input_size = static_cast<int>(normed.rows());
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.expert_ffn;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    Tensor gate_out = Tensor::gpu_scratch(s, scratch_key::kGate, {1, static_cast<int64_t>(ffn)});
    Tensor up_out = Tensor::gpu_scratch(s, scratch_key::kUp, {1, static_cast<int64_t>(ffn)});
    Tensor act = Tensor::gpu_scratch(s, scratch_key::kAct, {1, static_cast<int64_t>(ffn)});
    Tensor expert_out = Tensor::gpu_scratch(s, scratch_key::kExpertOut, {1, static_cast<int64_t>(hidden_size)});

    for (int tok = 0; tok < input_size; ++tok) {
        Tensor tok_in = Tensor::gpu_view(
            const_cast<float *>(d_normed + static_cast<size_t>(tok) * hidden_size),
            {1, static_cast<int64_t>(hidden_size)});
        for (int r = 0; r < k; ++r) {
            const size_t route_idx = static_cast<size_t>(tok) * k + r;
            const int e = route.expert_ids[route_idx];
            const float w = route.weights[route_idx];
            Tensor gate = expert_tensor_view(*lw_.ffn_gate_exps, e, n_exp);
            gate.shape = {static_cast<int64_t>(ffn), static_cast<int64_t>(hidden_size)};
            gate.to_gpu();
            gate.gemm(tok_in, gate_out, s, scratch_key::kFfnInLowp, "ds.gemm.egate");
            Tensor up = expert_tensor_view(*lw_.ffn_up_exps, e, n_exp);
            up.shape = {static_cast<int64_t>(ffn), static_cast<int64_t>(hidden_size)};
            up.to_gpu();
            up.gemm(tok_in, up_out, s, scratch_key::kFfnInLowp, "ds.gemm.eup");
            launch_silu_mul(gate_out.gpu_f32(), up_out.gpu_f32(), act.gpu_f32(), ffn, nullptr);
            Tensor down = expert_tensor_view(*lw_.ffn_down_exps, e, n_exp);
            down.shape = {static_cast<int64_t>(hidden_size), static_cast<int64_t>(ffn)};
            down.to_gpu();
            down.gemm(act, expert_out, s, scratch_key::kActLowp, "ds.gemm.edown");
            launch_moe_accumulate(expert_out.gpu_f32(), w, d_moe + static_cast<size_t>(tok) * hidden_size, hidden_size, nullptr);
        }
    }
}
