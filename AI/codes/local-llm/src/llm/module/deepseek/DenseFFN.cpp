//
// Created by zhangyoulun on 15/8/2026.
//

#include "DenseFFN.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <vector>

DenseFFN::DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), weights_(weights) {}

void DenseFFN::forward(DeepseekSession &session, const GPUTensor &g_hidden) {
    const int input_size = static_cast<int>(g_hidden.rows());
    auto &scratch = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};
    GPUTensor g_normed = GPUTensor(scratch, scratch_key::kNormed, act_shape, DType::F32);
    RMSNorm::forward(*weights_.s_ffn_norm, g_hidden, g_normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    const std::vector<int64_t> ffn_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(ffn)};
    GPUTensor g_gate = GPUTensor(scratch, scratch_key::kGate, ffn_shape, DType::F32);
    GPUTensor g_up = GPUTensor(scratch, scratch_key::kUp, ffn_shape, DType::F32);
    TensorTool::gemm(*weights_.s_ffn_gate, g_normed, g_gate, scratch, scratch_key::kFfnInLowp, "ds.gemm.d_ffn_gate");
    TensorTool::gemm(*weights_.s_ffn_up, g_normed, g_up, scratch, scratch_key::kFfnInLowp, "ds.gemm.d_ffn_up");

    GPUTensor g_act = GPUTensor(scratch, scratch_key::kAct, ffn_shape, DType::F32);
    TensorTool::silu_mul(g_gate, g_up, g_act);

    GPUTensor g_ffn_out = GPUTensor(scratch, scratch_key::kFfnOut, act_shape, DType::F32);
    TensorTool::gemm(*weights_.s_ffn_down, g_act, g_ffn_out, scratch, scratch_key::kActLowp, "ds.gemm.d_ffn_down");
    TensorTool::add(g_hidden, g_ffn_out, g_hidden);
}
