//
// Created by zhangyoulun on 15/8/2026.
//

#include "DenseFFN.h"

#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/deepseek/RMSNorm.h"

#include <cstddef>

DenseFFN::DenseFFN(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
                   deepseek::RMSNorm *rms_norm)
    : config_(config), weights_(weights), pool_(pool), rms_norm_(rms_norm) {}

void DenseFFN::forward(DeepseekSession &session, int layer, float *d_hidden, int tokens) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int H = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    float *d_normed = s.normed.ensure(static_cast<size_t>(tokens) * H, "ds.normed");
    rms_norm_->forward(*lw.ffn_norm, d_hidden, d_normed, tokens, H);

    float *d_gate = s.gate.ensure(static_cast<size_t>(tokens) * ffn, "ds.gate");
    float *d_up = s.up.ensure(static_cast<size_t>(tokens) * ffn, "ds.up");
    uint16_t *xlow = s.ffn_in_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_in_lowp");
    {
        CudaWeight wg = pool_->cached_weight(*lw.ffn_gate)->try_dequant();
        to_weight_lowp(d_normed, xlow, tokens * H, wg, nullptr);
        gemm_weight(pool_->handle, wg, ffn, H, xlow, wg.type, tokens, d_gate, "ds.gemm.ffn_gate");
    }
    {
        CudaWeight wu = pool_->cached_weight(*lw.ffn_up)->try_dequant();
        to_weight_lowp(d_normed, xlow, tokens * H, wu, nullptr);
        gemm_weight(pool_->handle, wu, ffn, H, xlow, wu.type, tokens, d_up, "ds.gemm.ffn_up");
    }
    float *d_act = s.act.ensure(static_cast<size_t>(tokens) * ffn, "ds.act");
    launch_silu_mul(d_gate, d_up, d_act, tokens * ffn, nullptr);

    float *d_out = s.ffn_out.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_out");
    {
        CudaWeight wd = pool_->cached_weight(*lw.ffn_down)->try_dequant();
        uint16_t *alow = s.act_lowp.ensure(static_cast<size_t>(tokens) * ffn, "ds.act_lowp");
        to_weight_lowp(d_act, alow, tokens * ffn, wd, nullptr);
        gemm_weight(pool_->handle, wd, H, ffn, alow, wd.type, tokens, d_out, "ds.gemm.ffn_down");
    }
    launch_add(d_hidden, d_out, d_hidden, tokens * H, nullptr);
}
