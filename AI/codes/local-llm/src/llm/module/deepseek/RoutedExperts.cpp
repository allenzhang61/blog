//
// Created by zhangyoulun on 15/8/2026.
//

#include "RoutedExperts.h"

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

RoutedExperts::RoutedExperts(const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config), pool_(pool) {}

void RoutedExperts::forward(DeepseekSession &session, const DeepseekLayerWeights &weights,
                            const float *d_normed, const MoERoute &route, int input_size, float *d_moe) {
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.expert_ffn;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;

    float *d_gate = s.gate.ensure(static_cast<size_t>(ffn), "ds.egate");
    float *d_up = s.up.ensure(static_cast<size_t>(ffn), "ds.eup");
    float *d_act = s.act.ensure(static_cast<size_t>(ffn), "ds.eact");
    float *d_eout = s.expert_out.ensure(static_cast<size_t>(hidden_size), "ds.eout");
    uint16_t *xlow = s.ffn_in_lowp.ensure(static_cast<size_t>(hidden_size), "ds.ffn_in_lowp");
    uint16_t *alow = s.act_lowp.ensure(static_cast<size_t>(ffn), "ds.act_lowp");

    for (int tok = 0; tok < input_size; ++tok) {
        const float *tok_in = d_normed + static_cast<size_t>(tok) * hidden_size;
        for (int r = 0; r < k; ++r) {
            const size_t route_idx = static_cast<size_t>(tok) * k + r;
            const int e = route.expert_ids[route_idx];
            const float w = route.weights[route_idx];
            MFTensorView gate = expert_tensor_view(*weights.ffn_gate_exps, e, n_exp);
            CudaWeight wg = pool_->cached_weight(gate)->try_dequant();
            GemmInput egate_in = prepare_gemm_input(tok_in, xlow, hidden_size, wg.type, nullptr);
            gemm_weight(pool_->handle, wg, egate_in.ptr, d_gate, ffn, hidden_size, 1, egate_in.type, "ds.gemm.egate");
            MFTensorView up = expert_tensor_view(*weights.ffn_up_exps, e, n_exp);
            CudaWeight wu = pool_->cached_weight(up)->try_dequant();
            GemmInput eup_in = prepare_gemm_input(tok_in, xlow, hidden_size, wu.type, nullptr);
            gemm_weight(pool_->handle, wu, eup_in.ptr, d_up, ffn, hidden_size, 1, eup_in.type, "ds.gemm.eup");
            launch_silu_mul(d_gate, d_up, d_act, ffn, nullptr);
            MFTensorView down = expert_tensor_view(*weights.ffn_down_exps, e, n_exp);
            CudaWeight wd = pool_->cached_weight(down)->try_dequant();
            GemmInput edown_in = prepare_gemm_input(d_act, alow, ffn, wd.type, nullptr);
            gemm_weight(pool_->handle, wd, edown_in.ptr, d_eout, hidden_size, ffn, 1, edown_in.type, "ds.gemm.edown");
            launch_moe_accumulate(d_eout, w, d_moe + static_cast<size_t>(tok) * hidden_size, hidden_size, nullptr);
        }
    }
}
