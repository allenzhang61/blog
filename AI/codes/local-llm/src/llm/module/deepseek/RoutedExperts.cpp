//
// Created by zhangyoulun on 15/8/2026.
//

#include "RoutedExperts.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "format/MF.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

int64_t tensor_elements(const MFTensorView &tensor) {
    int64_t n = 1;
    for (int64_t dim : tensor.shape) {
        n *= dim;
    }
    return n;
}

MFTensorView expert_tensor_view(const MFTensorView &weight, int expert, int n_experts) {
    const int64_t n_per_expert = tensor_elements(weight) / n_experts;
    const size_t bytes_per_expert = weight.nbytes / static_cast<size_t>(n_experts);

    MFTensorView view = weight;
    view.name = weight.name + ".e" + std::to_string(expert);
    view.shape = {n_per_expert};
    view.data = weight.data + static_cast<size_t>(expert) * bytes_per_expert;
    view.nbytes = bytes_per_expert;
    return view;
}

} // namespace

RoutedExperts::RoutedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config), lw_(weights), pool_(pool) {}

void RoutedExperts::forward(DeepseekSession &session, const float *d_normed, const MoERoute &route,
                            int input_size, float *d_moe) {
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.expert_ffn;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    float *d_gate = s.ensure<float>(scratch_key::kGate, static_cast<size_t>(ffn));
    float *d_up = s.ensure<float>(scratch_key::kUp, static_cast<size_t>(ffn));
    float *d_act = s.ensure<float>(scratch_key::kAct, static_cast<size_t>(ffn));
    float *d_expert_out = s.ensure<float>(scratch_key::kExpertOut, static_cast<size_t>(hidden_size));
    uint16_t *d_ffn_in_lowp = s.ensure<uint16_t>(scratch_key::kFfnInLowp, static_cast<size_t>(hidden_size));
    uint16_t *d_act_lowp = s.ensure<uint16_t>(scratch_key::kActLowp, static_cast<size_t>(ffn));

    for (int tok = 0; tok < input_size; ++tok) {
        const float *tok_in = d_normed + static_cast<size_t>(tok) * hidden_size;
        for (int r = 0; r < k; ++r) {
            const size_t route_idx = static_cast<size_t>(tok) * k + r;
            const int e = route.expert_ids[route_idx];
            const float w = route.weights[route_idx];
            MFTensorView gate = expert_tensor_view(*lw_.ffn_gate_exps, e, n_exp);
            CudaWeight ffn_gate_exps = pool_->cached_weight(gate)->try_dequant();
            GemmInput egate_in = prepare_gemm_input(tok_in, d_ffn_in_lowp, hidden_size, ffn_gate_exps.type, nullptr);
            gemm_weight(pool_->handle, ffn_gate_exps, egate_in.ptr, d_gate, ffn, hidden_size, 1, egate_in.type, "ds.gemm.egate");
            MFTensorView up = expert_tensor_view(*lw_.ffn_up_exps, e, n_exp);
            CudaWeight ffn_up_exps = pool_->cached_weight(up)->try_dequant();
            GemmInput eup_in = prepare_gemm_input(tok_in, d_ffn_in_lowp, hidden_size, ffn_up_exps.type, nullptr);
            gemm_weight(pool_->handle, ffn_up_exps, eup_in.ptr, d_up, ffn, hidden_size, 1, eup_in.type, "ds.gemm.eup");
            launch_silu_mul(d_gate, d_up, d_act, ffn, nullptr);
            MFTensorView down = expert_tensor_view(*lw_.ffn_down_exps, e, n_exp);
            CudaWeight ffn_down_exps = pool_->cached_weight(down)->try_dequant();
            GemmInput edown_in = prepare_gemm_input(d_act, d_act_lowp, ffn, ffn_down_exps.type, nullptr);
            gemm_weight(pool_->handle, ffn_down_exps, edown_in.ptr, d_expert_out, hidden_size, ffn, 1, edown_in.type, "ds.gemm.edown");
            launch_moe_accumulate(d_expert_out, w, d_moe + static_cast<size_t>(tok) * hidden_size, hidden_size, nullptr);
        }
    }
}
