//
// Created by zhangyoulun on 15/8/2026.
//

#include "DenseFFN.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"

#include <cstddef>
#include <vector>

DenseFFN::DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), weights_(weights) {}

void DenseFFN::forward(DeepseekSession &session, const Tensor &hidden) {
    float *d_hidden = hidden.gpu_f32();
    const int input_size = static_cast<int>(hidden.rows());
    auto &scratch = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};
    Tensor normed = Tensor::gpu_scratch(scratch, scratch_key::kNormed, act_shape);
    RMSNorm::forward(*weights_.ffn_norm, hidden, normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    const std::vector<int64_t> ffn_shape = {static_cast<int64_t>(input_size), static_cast<int64_t>(ffn)};
    Tensor gate = Tensor::gpu_scratch(scratch, scratch_key::kGate, ffn_shape);
    Tensor up = Tensor::gpu_scratch(scratch, scratch_key::kUp, ffn_shape);
    weights_.ffn_gate->to_gpu();
    weights_.ffn_gate->gemm(normed, gate, scratch, scratch_key::kFfnInLowp, "ds.gemm.ffn_gate");
    weights_.ffn_up->to_gpu();
    weights_.ffn_up->gemm(normed, up, scratch, scratch_key::kFfnInLowp, "ds.gemm.ffn_up");

    Tensor act = Tensor::gpu_scratch(scratch, scratch_key::kAct, ffn_shape);
    launch_silu_mul(gate.gpu_f32(), up.gpu_f32(), act.gpu_f32(), input_size * ffn, nullptr);

    Tensor ffn_out = Tensor::gpu_scratch(scratch, scratch_key::kFfnOut, act_shape);
    weights_.ffn_down->to_gpu();
    weights_.ffn_down->gemm(act, ffn_out, scratch, scratch_key::kActLowp, "ds.gemm.ffn_down");
    launch_add(d_hidden, ffn_out.gpu_f32(), d_hidden, input_size * hidden_size, nullptr);
}
