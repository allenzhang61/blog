//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLA.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"

#include <cuda_runtime.h>

MLA::MLA(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool)
    : config_(config), weights_(weights), pool_(pool) {}

void MLA::forward(DeepseekSession &session, int layer, float *d_hidden, int input_size, int start_pos) {
    auto &scratch = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int hidden_size = config_.hidden_size;
    const int n_heads = config_.num_heads;
    const int qk_nope = config_.qk_nope_head_dim;
    const int qk_rope = config_.qk_rope_head_dim;
    const int qk_head = config_.qk_head_dim();       // 192
    const int v_head = config_.v_head_dim;           // 128
    const int kv_lora = config_.kv_lora_rank;        // 512
    const int kv_total = kv_lora + qk_rope;          // 576
    const int q_dim = n_heads * qk_head;             // 3072
    const int kvb_out = n_heads * (qk_nope + v_head); // 4096

    float *d_normed = scratch.normed.ensure(static_cast<size_t>(input_size) * hidden_size, "ds.normed");
    RMSNorm::forward(pool_, *lw.attn_norm, d_hidden, d_normed, input_size, hidden_size,
                     config_.rms_norm_eps, /*one_plus=*/false);

    float *d_q = scratch.q.ensure(static_cast<size_t>(input_size) * q_dim, "ds.q");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_q)->try_dequant();
        uint16_t *xlow = scratch.normed_lowp.ensure(static_cast<size_t>(input_size) * hidden_size, "ds.normed_lowp");
        GemmInput q_in = prepare_gemm_input(d_normed, xlow, input_size * hidden_size, w.type, nullptr);
        gemm_weight(pool_->handle, w, q_in.ptr, d_q, q_dim, hidden_size, input_size, q_in.type, "ds.gemm.attn_q");
    }
    if (input_size == 1) {
        launch_mla_rope_q(d_q, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    } else {
        launch_mla_rope_q_batch(d_q, input_size, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    }

    float *d_kv_a = scratch.kv_a.ensure(static_cast<size_t>(input_size) * kv_total, "ds.kv_a");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_kv_a_mqa)->try_dequant();
        uint16_t *xlow = scratch.normed_lowp.ensure(static_cast<size_t>(input_size) * hidden_size, "ds.normed_lowp");
        GemmInput kv_a_in = prepare_gemm_input(d_normed, xlow, input_size * hidden_size, w.type, nullptr);
        gemm_weight(pool_->handle, w, kv_a_in.ptr, d_kv_a, kv_total, hidden_size, input_size, kv_a_in.type, "ds.gemm.kv_a");
    }

    float *d_cache = static_cast<float *>(session.kv_caches[layer].cache.ptr);
    CudaWeight *kv_a_norm_weight = pool_->cached_weight(*lw.attn_kv_a_norm);
    const float *kv_a_norm = static_cast<const float *>(kv_a_norm_weight->ptr);
    if (input_size == 1) {
        launch_mla_kv_a(d_kv_a, kv_a_norm, d_cache, kv_lora, qk_rope, session.max_seq_len,
                        start_pos, static_cast<const float *>(session.inv_freq.ptr),
                        config_.rms_norm_eps, nullptr);
    } else {
        launch_mla_kv_a_batch(d_kv_a, kv_a_norm, d_cache, input_size, kv_lora, qk_rope,
                              session.max_seq_len, start_pos,
                              static_cast<const float *>(session.inv_freq.ptr),
                              config_.rms_norm_eps, nullptr);
    }
    session.kv_caches[layer].seq_len = start_pos + input_size;

    const int seq = start_pos + input_size;
    float *d_kvb = scratch.kv_b_out.ensure(static_cast<size_t>(seq) * kvb_out, "ds.kv_b_out");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_kv_b)->try_dequant();
        uint16_t *latent_low = scratch.latent_lowp.ensure(static_cast<size_t>(seq) * kv_lora, "ds.latent_lowp");
        float *latent_f32 = scratch.attn.ensure(static_cast<size_t>(seq) * kv_lora, "ds.latent_f32");
        check_cuda(cudaMemcpy2D(latent_f32, kv_lora * sizeof(float), d_cache,
                                kv_total * sizeof(float), kv_lora * sizeof(float), seq,
                                cudaMemcpyDeviceToDevice),
                   "ds.gather.latent");
        GemmInput latent_in = prepare_gemm_input(latent_f32, latent_low, seq * kv_lora, w.type, nullptr);
        gemm_weight(pool_->handle, w, latent_in.ptr, d_kvb, kvb_out, kv_lora, seq, latent_in.type, "ds.gemm.kv_b");
    }

    float *d_attn = scratch.attn.ensure(static_cast<size_t>(input_size) * n_heads * v_head, "ds.attn");
    if (input_size == 1) {
        launch_mla_attend(d_q, d_kvb, d_cache, d_attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                          session.max_seq_len, start_pos, session.attn_softmax_scale, nullptr);
    } else {
        launch_mla_attend_batch(d_q, d_kvb, d_cache, d_attn, input_size, n_heads, qk_nope, qk_rope,
                                v_head, kv_lora, session.max_seq_len, start_pos,
                                session.attn_softmax_scale, nullptr);
    }

    float *d_out = scratch.attn_out.ensure(static_cast<size_t>(input_size) * hidden_size, "ds.attn_out");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_output)->try_dequant();
        const int in_dim = n_heads * v_head; // 2048
        uint16_t *xlow = scratch.attn_lowp.ensure(static_cast<size_t>(input_size) * in_dim, "ds.attn_lowp");
        GemmInput attn_in = prepare_gemm_input(d_attn, xlow, input_size * in_dim, w.type, nullptr);
        gemm_weight(pool_->handle, w, attn_in.ptr, d_out, hidden_size, in_dim, input_size, attn_in.type, "ds.gemm.attn_output");
    }

    launch_add(d_hidden, d_out, d_hidden, input_size * hidden_size, nullptr);
}
