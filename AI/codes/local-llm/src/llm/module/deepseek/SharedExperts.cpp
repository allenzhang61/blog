//
// Created by zhangyoulun on 15/8/2026.
//

#include "SharedExperts.h"

#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"

#include <cstddef>

SharedExperts::SharedExperts(const DeepseekConfig &config, CudaWeightPool *pool)
    : config_(config), pool_(pool) {}

void SharedExperts::forward(DeepseekSession &session, const DeepseekLayerWeights &weights,
                            const float *d_normed, int tokens, float *d_moe) {
    auto &s = session.scratch;
    const int H = config_.hidden_size;
    const int shared_ffn = config_.shared_ffn();

    float *d_sgate = s.gate.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.sgate");
    float *d_sup = s.up.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.sup");
    uint16_t *sxlow = s.ffn_in_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_in_lowp");
    CudaWeight wg = pool_->cached_weight(*weights.ffn_gate_shexp)->try_dequant();
    to_weight_lowp(d_normed, sxlow, tokens * H, wg, nullptr);
    gemm_weight(pool_->handle, wg, shared_ffn, H, sxlow, wg.type, tokens, d_sgate, "ds.gemm.sgate");
    CudaWeight wu = pool_->cached_weight(*weights.ffn_up_shexp)->try_dequant();
    to_weight_lowp(d_normed, sxlow, tokens * H, wu, nullptr);
    gemm_weight(pool_->handle, wu, shared_ffn, H, sxlow, wu.type, tokens, d_sup, "ds.gemm.sup");
    float *d_sact = s.act.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.sact");
    launch_silu_mul(d_sgate, d_sup, d_sact, tokens * shared_ffn, nullptr);
    float *d_sout = s.ffn_out.ensure(static_cast<size_t>(tokens) * H, "ds.sout");
    CudaWeight wd = pool_->cached_weight(*weights.ffn_down_shexp)->try_dequant();
    uint16_t *salow = s.act_lowp.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.act_lowp");
    to_weight_lowp(d_sact, salow, tokens * shared_ffn, wd, nullptr);
    gemm_weight(pool_->handle, wd, H, shared_ffn, salow, wd.type, tokens, d_sout, "ds.gemm.sdown");
    launch_add(d_moe, d_sout, d_moe, tokens * H, nullptr);
}
