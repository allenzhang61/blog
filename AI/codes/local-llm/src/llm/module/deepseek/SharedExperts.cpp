//
// Created by zhangyoulun on 15/8/2026.
//

#include "SharedExperts.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <vector>

SharedExperts::SharedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

void SharedExperts::forward(DeepseekSession &session, const Tensor &normed, const Tensor &moe) {
    const int input_size = static_cast<int>(normed.rows());
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int shared_ffn = config_.shared_ffn();

    const std::vector<int64_t> shared_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(shared_ffn)};
    Tensor gate = Tensor::gpu_scratch(s, scratch_key::kGate, shared_shape);
    Tensor up = Tensor::gpu_scratch(s, scratch_key::kUp, shared_shape);
    TensorTool::gemm(*lw_.ffn_gate_shexp, normed, gate, s, scratch_key::kFfnInLowp, "ds.gemm.sgate");
    TensorTool::gemm(*lw_.ffn_up_shexp, normed, up, s, scratch_key::kFfnInLowp, "ds.gemm.sup");

    Tensor act = Tensor::gpu_scratch(s, scratch_key::kAct, shared_shape);
    TensorTool::silu_mul(gate, up, act);

    Tensor ffn_out = Tensor::gpu_scratch(
        s, scratch_key::kFfnOut, {static_cast<int64_t>(input_size), static_cast<int64_t>(hidden_size)});
    TensorTool::gemm(*lw_.ffn_down_shexp, act, ffn_out, s, scratch_key::kActLowp, "ds.gemm.sdown");
    TensorTool::add(moe, ffn_out, moe);
}
