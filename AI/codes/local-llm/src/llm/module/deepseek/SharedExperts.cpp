//
// Created by zhangyoulun on 15/8/2026.
//

#include "SharedExperts.h"

#include "backend/cuda/mem/CudaScratch.h"
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
                            const float *d_normed, int input_size, float *d_moe) {
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int shared_ffn = config_.shared_ffn();

    float *d_sgate = s.ensure<float>(scratch_key::kGate, static_cast<size_t>(input_size) * shared_ffn, "ds.sgate");
    float *d_sup = s.ensure<float>(scratch_key::kUp, static_cast<size_t>(input_size) * shared_ffn, "ds.sup");
    uint16_t *sxlow = s.ensure<uint16_t>(scratch_key::kFfnInLowp, static_cast<size_t>(input_size) * hidden_size, "ds.ffn_in_lowp");
    CudaWeight wg = pool_->cached_weight(*weights.ffn_gate_shexp)->try_dequant();
    GemmInput sgate_in = prepare_gemm_input(d_normed, sxlow, input_size * hidden_size, wg.type, nullptr);
    gemm_weight(pool_->handle, wg, sgate_in.ptr, d_sgate, shared_ffn, hidden_size, input_size, sgate_in.type, "ds.gemm.sgate");
    CudaWeight wu = pool_->cached_weight(*weights.ffn_up_shexp)->try_dequant();
    GemmInput sup_in = prepare_gemm_input(d_normed, sxlow, input_size * hidden_size, wu.type, nullptr);
    gemm_weight(pool_->handle, wu, sup_in.ptr, d_sup, shared_ffn, hidden_size, input_size, sup_in.type, "ds.gemm.sup");
    float *d_sact = s.ensure<float>(scratch_key::kAct, static_cast<size_t>(input_size) * shared_ffn, "ds.sact");
    launch_silu_mul(d_sgate, d_sup, d_sact, input_size * shared_ffn, nullptr);
    float *d_sout = s.ensure<float>(scratch_key::kFfnOut, static_cast<size_t>(input_size) * hidden_size, "ds.sout");
    CudaWeight wd = pool_->cached_weight(*weights.ffn_down_shexp)->try_dequant();
    uint16_t *salow = s.ensure<uint16_t>(scratch_key::kActLowp, static_cast<size_t>(input_size) * shared_ffn, "ds.act_lowp");
    GemmInput sdown_in = prepare_gemm_input(d_sact, salow, input_size * shared_ffn, wd.type, nullptr);
    gemm_weight(pool_->handle, wd, sdown_in.ptr, d_sout, hidden_size, shared_ffn, input_size, sdown_in.type, "ds.gemm.sdown");
    launch_add(d_moe, d_sout, d_moe, input_size * hidden_size, nullptr);
}
