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

SharedExperts::SharedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

void SharedExperts::forward(DeepseekSession &session, const Tensor &normed, const Tensor &moe) {
    const float *d_normed = normed.gpu_f32();
    float *d_moe = moe.gpu_f32();
    const int input_size = static_cast<int>(normed.rows());
    auto &s = session.scratch;
    const int hidden_size = config_.hidden_size;
    const int shared_ffn = config_.shared_ffn();

    float *d_gate = s.ensure<float>(scratch_key::kGate, static_cast<size_t>(input_size) * shared_ffn);
    float *d_up = s.ensure<float>(scratch_key::kUp, static_cast<size_t>(input_size) * shared_ffn);
    uint16_t *d_ffn_in_lowp = s.ensure<uint16_t>(scratch_key::kFfnInLowp, static_cast<size_t>(input_size) * hidden_size);
    CudaWeight ffn_gate_shexp = lw_.ffn_gate_shexp->cached_weight()->try_dequant();
    GemmInput sgate_in = prepare_gemm_input(d_normed, d_ffn_in_lowp, input_size * hidden_size, ffn_gate_shexp.type, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, ffn_gate_shexp, sgate_in.ptr, d_gate, shared_ffn, hidden_size, input_size, sgate_in.type, "ds.gemm.sgate");
    CudaWeight ffn_up_shexp = lw_.ffn_up_shexp->cached_weight()->try_dequant();
    GemmInput sup_in = prepare_gemm_input(d_normed, d_ffn_in_lowp, input_size * hidden_size, ffn_up_shexp.type, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, ffn_up_shexp, sup_in.ptr, d_up, shared_ffn, hidden_size, input_size, sup_in.type, "ds.gemm.sup");
    float *d_act = s.ensure<float>(scratch_key::kAct, static_cast<size_t>(input_size) * shared_ffn);
    launch_silu_mul(d_gate, d_up, d_act, input_size * shared_ffn, nullptr);
    float *d_ffn_out = s.ensure<float>(scratch_key::kFfnOut, static_cast<size_t>(input_size) * hidden_size);
    CudaWeight ffn_down_shexp = lw_.ffn_down_shexp->cached_weight()->try_dequant();
    uint16_t *d_act_lowp = s.ensure<uint16_t>(scratch_key::kActLowp, static_cast<size_t>(input_size) * shared_ffn);
    GemmInput sdown_in = prepare_gemm_input(d_act, d_act_lowp, input_size * shared_ffn, ffn_down_shexp.type, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, ffn_down_shexp, sdown_in.ptr, d_ffn_out, hidden_size, shared_ffn, input_size, sdown_in.type, "ds.gemm.sdown");
    launch_add(d_moe, d_ffn_out, d_moe, input_size * hidden_size, nullptr);
}
