//
// Created by zhangyoulun on 15/8/2026.
//

#include "DenseFFN.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"

#include <cstddef>

DenseFFN::DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config), weights_(weights), pool_(pool) {}

void DenseFFN::forward(DeepseekSession &session, float *d_hidden, int input_size) {
    auto &scratch = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    float *d_normed = scratch.ensure<float>(scratch_key::kNormed, static_cast<size_t>(input_size) * hidden_size);
    RMSNorm::forward(pool_, *weights_.ffn_norm, d_hidden, d_normed, input_size, hidden_size,
                     config_.rms_norm_eps, /*one_plus=*/false);

    float *d_gate = scratch.ensure<float>(scratch_key::kGate, static_cast<size_t>(input_size) * ffn);
    float *d_up = scratch.ensure<float>(scratch_key::kUp, static_cast<size_t>(input_size) * ffn);
    uint16_t *d_ffn_in_lowp = scratch.ensure<uint16_t>(scratch_key::kFfnInLowp, static_cast<size_t>(input_size) * hidden_size);
    {
        CudaWeight ffn_gate = pool_->cached_weight(*weights_.ffn_gate)->try_dequant();
        GemmInput gate_in = prepare_gemm_input(d_normed, d_ffn_in_lowp, input_size * hidden_size, ffn_gate.type, nullptr);
        gemm_weight(pool_->handle, ffn_gate, gate_in.ptr, d_gate, ffn, hidden_size, input_size, gate_in.type, "ds.gemm.ffn_gate");
    }
    {
        CudaWeight ffn_up = pool_->cached_weight(*weights_.ffn_up)->try_dequant();
        GemmInput up_in = prepare_gemm_input(d_normed, d_ffn_in_lowp, input_size * hidden_size, ffn_up.type, nullptr);
        gemm_weight(pool_->handle, ffn_up, up_in.ptr, d_up, ffn, hidden_size, input_size, up_in.type, "ds.gemm.ffn_up");
    }
    float *d_act = scratch.ensure<float>(scratch_key::kAct, static_cast<size_t>(input_size) * ffn);
    launch_silu_mul(d_gate, d_up, d_act, input_size * ffn, nullptr);

    float *d_ffn_out = scratch.ensure<float>(scratch_key::kFfnOut, static_cast<size_t>(input_size) * hidden_size);
    {
        CudaWeight ffn_down = pool_->cached_weight(*weights_.ffn_down)->try_dequant();
        uint16_t *d_act_lowp = scratch.ensure<uint16_t>(scratch_key::kActLowp, static_cast<size_t>(input_size) * ffn);
        GemmInput down_in = prepare_gemm_input(d_act, d_act_lowp, input_size * ffn, ffn_down.type, nullptr);
        gemm_weight(pool_->handle, ffn_down, down_in.ptr, d_ffn_out, hidden_size, ffn, input_size, down_in.type, "ds.gemm.ffn_down");
    }
    launch_add(d_hidden, d_ffn_out, d_hidden, input_size * hidden_size, nullptr);
}
