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
#include <utility>
#include <vector>

namespace {
    StorageTensor s_expert_tensor_view(const StorageTensor &s_weight, int expert, int n_experts,
                                       std::vector<int64_t> shape) {
        const int64_t n_per_expert = s_weight.numel() / n_experts;
        const size_t bytes_per_expert = s_weight.nbytes / static_cast<size_t>(n_experts);

        return s_weight.slice(static_cast<size_t>(expert) * bytes_per_expert,
                              shape.empty() ? std::vector<int64_t>{n_per_expert} : std::move(shape),
                              bytes_per_expert,
                              s_weight.name + ".e" + std::to_string(expert));
    }

} // namespace

RoutedExperts::RoutedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {
    if (!lw_.is_moe) {
        return;
    }
    const int n_exp = config_.expert_count;
    const int64_t hidden_size = config_.hidden_size;
    const int64_t ffn = config_.expert_ffn;
    s_gate_experts_.reserve(static_cast<size_t>(n_exp));
    s_up_experts_.reserve(static_cast<size_t>(n_exp));
    s_down_experts_.reserve(static_cast<size_t>(n_exp));
    for (int e = 0; e < n_exp; ++e) {
        s_gate_experts_.push_back(s_expert_tensor_view(*lw_.s_ffn_gate_exps, e, n_exp, {ffn, hidden_size}));
        s_up_experts_.push_back(s_expert_tensor_view(*lw_.s_ffn_up_exps, e, n_exp, {ffn, hidden_size}));
        s_down_experts_.push_back(s_expert_tensor_view(*lw_.s_ffn_down_exps, e, n_exp, {hidden_size, ffn}));
    }
}

void RoutedExperts::forward(DeepseekSession &session, const GPUTensor &g_normed_f32, const MoERoute &route,
                            const GPUTensor &g_moe_f32) {
    const int64_t input_size = g_normed_f32.rows();
    auto &s = session.cuda_scratch;
    const int64_t hidden_size = config_.hidden_size;
    const int64_t ffn = config_.expert_ffn;
    const int k = config_.expert_used;
    const int *expert_ids = route.c_expert_ids_i32.data<int>();
    // decode：top_w 留在 device（route.decode_device），加权累加从 device 读权重，省掉每层 top_w 的回读同步。
    const bool decode_device = (input_size == 1 && route.decode_device);
    const float *weights = decode_device ? nullptr : route.c_weights_f32.data<float>();

    GPUTensor g_gate_out_f32 = GPUTensor(s, scratch_key::kGate, {1, ffn}, DType::F32);
    GPUTensor g_up_out_f32 = GPUTensor(s, scratch_key::kUp, {1, ffn}, DType::F32);
    GPUTensor g_act_f32 = GPUTensor(s, scratch_key::kAct, {1, ffn}, DType::F32);
    GPUTensor g_expert_out_f32 = GPUTensor(
        s, scratch_key::kExpertOut, {1, hidden_size}, DType::F32);

    for (int64_t input_index = 0; input_index < input_size; ++input_index) {
        const size_t token_offset = static_cast<size_t>(input_index) * hidden_size * sizeof(float);
        GPUTensor g_tok_in_f32 = GPUTensor(g_normed_f32, token_offset, {1, hidden_size});
        for (int r = 0; r < k; ++r) {
            const size_t route_idx = static_cast<size_t>(input_index) * k + r;
            const int e = expert_ids[route_idx];
            TensorTool::gemm(s_gate_experts_[e], g_tok_in_f32, g_gate_out_f32, s, scratch_key::kFfnInLowp,
                             "ds.gemm.egate");
            TensorTool::gemm(s_up_experts_[e], g_tok_in_f32, g_up_out_f32, s, scratch_key::kFfnInLowp,
                             "ds.gemm.eup");
            TensorTool::silu_mul(g_gate_out_f32, g_up_out_f32, g_act_f32);
            TensorTool::gemm(s_down_experts_[e], g_act_f32, g_expert_out_f32, s, scratch_key::kActLowp,
                             "ds.gemm.edown");
            GPUTensor g_tok_moe_f32 = GPUTensor(g_moe_f32, token_offset, {1, hidden_size});
            if (decode_device) {
                // 权重指针指向 device 端 top_w[r]，avoid host readback。
                TensorTool::moe_accumulate_device(g_expert_out_f32, route.d_weights_f32 + r, g_tok_moe_f32);
            } else {
                TensorTool::moe_accumulate(g_expert_out_f32, weights[route_idx], g_tok_moe_f32);
            }
        }
    }
}
