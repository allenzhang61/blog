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

void DenseFFN::forward(DeepseekSession &session, const GPUTensor &g_hidden_f32) {
    const int input_size = static_cast<int>(g_hidden_f32.rows());
    auto &scratch = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};
    GPUTensor g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, act_shape, DType::F32);
    RMSNorm::forward(*weights_.s_ffn_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);

    const std::vector<int64_t> ffn_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(ffn)};
    GPUTensor g_gate_f32 = GPUTensor(scratch, scratch_key::kGate, ffn_shape, DType::F32);
    GPUTensor g_up_f32 = GPUTensor(scratch, scratch_key::kUp, ffn_shape, DType::F32);
    TensorTool::gemm(*weights_.s_ffn_gate, g_normed_f32, g_gate_f32, scratch, scratch_key::kFfnInLowp, "ds.gemm.d_ffn_gate");
    TensorTool::gemm(*weights_.s_ffn_up, g_normed_f32, g_up_f32, scratch, scratch_key::kFfnInLowp, "ds.gemm.d_ffn_up");

    GPUTensor g_act_f32 = GPUTensor(scratch, scratch_key::kAct, ffn_shape, DType::F32);
    TensorTool::silu_mul(g_gate_f32, g_up_f32, g_act_f32);

    GPUTensor g_ffn_out_f32 = GPUTensor(scratch, scratch_key::kFfnOut, act_shape, DType::F32);
    TensorTool::gemm(*weights_.s_ffn_down, g_act_f32, g_ffn_out_f32, scratch, scratch_key::kActLowp, "ds.gemm.d_ffn_down");
    TensorTool::add(g_hidden_f32, g_ffn_out_f32, g_hidden_f32);
}
