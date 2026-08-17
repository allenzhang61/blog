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

DenseFFN::DenseFFN(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool)
    : config_(config), weights_(weights), pool_(pool) {}

void DenseFFN::forward(DeepseekSession &session, int layer, float *d_hidden, int input_size) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int hidden_size = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    float *d_normed = s.ensure<float>(scratch_key::kNormed, static_cast<size_t>(input_size) * hidden_size, "ds.normed");
    RMSNorm::forward(pool_, *lw.ffn_norm, d_hidden, d_normed, input_size, hidden_size,
                     config_.rms_norm_eps, /*one_plus=*/false);

    float *d_gate = s.ensure<float>(scratch_key::kGate, static_cast<size_t>(input_size) * ffn, "ds.gate");
    float *d_up = s.ensure<float>(scratch_key::kUp, static_cast<size_t>(input_size) * ffn, "ds.up");
    uint16_t *xlow = s.ensure<uint16_t>(scratch_key::kFfnInLowp, static_cast<size_t>(input_size) * hidden_size, "ds.ffn_in_lowp");
    {
        CudaWeight wg = pool_->cached_weight(*lw.ffn_gate)->try_dequant();
        GemmInput gate_in = prepare_gemm_input(d_normed, xlow, input_size * hidden_size, wg.type, nullptr);
        gemm_weight(pool_->handle, wg, gate_in.ptr, d_gate, ffn, hidden_size, input_size, gate_in.type, "ds.gemm.ffn_gate");
    }
    {
        CudaWeight wu = pool_->cached_weight(*lw.ffn_up)->try_dequant();
        GemmInput up_in = prepare_gemm_input(d_normed, xlow, input_size * hidden_size, wu.type, nullptr);
        gemm_weight(pool_->handle, wu, up_in.ptr, d_up, ffn, hidden_size, input_size, up_in.type, "ds.gemm.ffn_up");
    }
    float *d_act = s.ensure<float>(scratch_key::kAct, static_cast<size_t>(input_size) * ffn, "ds.act");
    launch_silu_mul(d_gate, d_up, d_act, input_size * ffn, nullptr);

    float *d_out = s.ensure<float>(scratch_key::kFfnOut, static_cast<size_t>(input_size) * hidden_size, "ds.ffn_out");
    {
        CudaWeight wd = pool_->cached_weight(*lw.ffn_down)->try_dequant();
        uint16_t *alow = s.ensure<uint16_t>(scratch_key::kActLowp, static_cast<size_t>(input_size) * ffn, "ds.act_lowp");
        GemmInput down_in = prepare_gemm_input(d_act, alow, input_size * ffn, wd.type, nullptr);
        gemm_weight(pool_->handle, wd, down_in.ptr, d_out, hidden_size, ffn, input_size, down_in.type, "ds.gemm.ffn_down");
    }
    launch_add(d_hidden, d_out, d_hidden, input_size * hidden_size, nullptr);
}
