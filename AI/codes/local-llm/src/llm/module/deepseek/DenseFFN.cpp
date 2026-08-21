//
// Created by zhangyoulun on 15/8/2026.
//

#include "DenseFFN.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/TensorTool.h"

#include <cstddef>
#include <vector>

DenseFFN::DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), weights_(weights) {}

void DenseFFN::forward(DeepseekSession &session, const GPUTensor &hidden) {
    const int input_size = static_cast<int>(hidden.rows());
    auto &scratch = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};
    GPUTensor normed = GPUTensor::gpu_scratch(scratch, scratch_key::kNormed, act_shape);
    RMSNorm::forward(*weights_.ffn_norm, hidden, normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    const std::vector<int64_t> ffn_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(ffn)};
    GPUTensor gate = GPUTensor::gpu_scratch(scratch, scratch_key::kGate, ffn_shape);
    GPUTensor up = GPUTensor::gpu_scratch(scratch, scratch_key::kUp, ffn_shape);
    TensorTool::gemm(*weights_.ffn_gate, normed, gate, scratch, scratch_key::kFfnInLowp, "ds.gemm.ffn_gate");
    TensorTool::gemm(*weights_.ffn_up, normed, up, scratch, scratch_key::kFfnInLowp, "ds.gemm.ffn_up");

    GPUTensor act = GPUTensor::gpu_scratch(scratch, scratch_key::kAct, ffn_shape);
    TensorTool::silu_mul(gate, up, act);

    GPUTensor ffn_out = GPUTensor::gpu_scratch(scratch, scratch_key::kFfnOut, act_shape);
    TensorTool::gemm(*weights_.ffn_down, act, ffn_out, scratch, scratch_key::kActLowp, "ds.gemm.ffn_down");
    TensorTool::add(hidden, ffn_out, hidden);
}
