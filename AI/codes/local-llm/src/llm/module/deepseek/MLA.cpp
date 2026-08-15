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

#include <cuda_runtime.h>

MLA::MLA(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool)
    : config_(config), weights_(weights), pool_(pool) {}

void MLA::forward(DeepseekSession &session, int layer, int tokens, int start_pos) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int H = config_.hidden_size;
    const int n_heads = config_.num_heads;
    const int qk_nope = config_.qk_nope_head_dim;
    const int qk_rope = config_.qk_rope_head_dim;
    const int qk_head = config_.qk_head_dim();       // 192
    const int v_head = config_.v_head_dim;           // 128
    const int kv_lora = config_.kv_lora_rank;        // 512
    const int kv_total = kv_lora + qk_rope;          // 576
    const int q_dim = n_heads * qk_head;             // 3072
    const int kvb_out = n_heads * (qk_nope + v_head); // 4096

    float *d_normed = s.normed.ensure(static_cast<size_t>(tokens) * H, "ds.normed");
    CudaWeight *attn_norm = pool_->cached_weight(*lw.attn_norm);
    launch_rms_norm(s.hidden, attn_norm->ptr, 2, d_normed, tokens, H,
                    config_.rms_norm_eps, false, nullptr);

    float *d_q = s.q.ensure(static_cast<size_t>(tokens) * q_dim, "ds.q");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_q)->try_dequant();
        uint16_t *xlow = s.normed_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.normed_lowp");
        to_weight_lowp(d_normed, xlow, tokens * H, w, nullptr);
        gemm_weight(pool_->handle, w, q_dim, H, xlow, w.type, tokens, d_q, "ds.gemm.attn_q");
    }
    if (tokens == 1) {
        launch_mla_rope_q(d_q, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    } else {
        launch_mla_rope_q_batch(d_q, tokens, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    }

    float *d_kv_a = s.kv_a.ensure(static_cast<size_t>(tokens) * kv_total, "ds.kv_a");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_kv_a_mqa)->try_dequant();
        uint16_t *xlow = s.normed_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.normed_lowp");
        to_weight_lowp(d_normed, xlow, tokens * H, w, nullptr);
        gemm_weight(pool_->handle, w, kv_total, H, xlow, w.type, tokens, d_kv_a, "ds.gemm.kv_a");
    }

    float *d_cache = static_cast<float *>(session.kv_caches[layer].cache.ptr);
    CudaWeight *kv_a_norm_weight = pool_->cached_weight(*lw.attn_kv_a_norm);
    const float *kv_a_norm = static_cast<const float *>(kv_a_norm_weight->ptr);
    if (tokens == 1) {
        launch_mla_kv_a(d_kv_a, kv_a_norm, d_cache, kv_lora, qk_rope, session.max_seq_len,
                        start_pos, static_cast<const float *>(session.inv_freq.ptr),
                        config_.rms_norm_eps, nullptr);
    } else {
        launch_mla_kv_a_batch(d_kv_a, kv_a_norm, d_cache, tokens, kv_lora, qk_rope,
                              session.max_seq_len, start_pos,
                              static_cast<const float *>(session.inv_freq.ptr),
                              config_.rms_norm_eps, nullptr);
    }
    session.kv_caches[layer].seq_len = start_pos + tokens;

    const int seq = start_pos + tokens;
    float *d_kvb = s.kv_b_out.ensure(static_cast<size_t>(seq) * kvb_out, "ds.kv_b_out");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_kv_b)->try_dequant();
        uint16_t *latent_low = s.latent_lowp.ensure(static_cast<size_t>(seq) * kv_lora, "ds.latent_lowp");
        float *latent_f32 = s.attn.ensure(static_cast<size_t>(seq) * kv_lora, "ds.latent_f32");
        check_cuda(cudaMemcpy2D(latent_f32, kv_lora * sizeof(float), d_cache,
                                kv_total * sizeof(float), kv_lora * sizeof(float), seq,
                                cudaMemcpyDeviceToDevice),
                   "ds.gather.latent");
        to_weight_lowp(latent_f32, latent_low, seq * kv_lora, w, nullptr);
        gemm_weight(pool_->handle, w, kvb_out, kv_lora, latent_low, w.type, seq, d_kvb, "ds.gemm.kv_b");
    }

    float *d_attn = s.attn.ensure(static_cast<size_t>(tokens) * n_heads * v_head, "ds.attn");
    if (tokens == 1) {
        launch_mla_attend(d_q, d_kvb, d_cache, d_attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                          session.max_seq_len, start_pos, session.attn_softmax_scale, nullptr);
    } else {
        launch_mla_attend_batch(d_q, d_kvb, d_cache, d_attn, tokens, n_heads, qk_nope, qk_rope,
                                v_head, kv_lora, session.max_seq_len, start_pos,
                                session.attn_softmax_scale, nullptr);
    }

    float *d_out = s.attn_out.ensure(static_cast<size_t>(tokens) * H, "ds.attn_out");
    {
        CudaWeight w = pool_->cached_weight(*lw.attn_output)->try_dequant();
        const int in_dim = n_heads * v_head; // 2048
        uint16_t *xlow = s.attn_lowp.ensure(static_cast<size_t>(tokens) * in_dim, "ds.attn_lowp");
        to_weight_lowp(d_attn, xlow, tokens * in_dim, w, nullptr);
        gemm_weight(pool_->handle, w, H, in_dim, xlow, w.type, tokens, d_out, "ds.gemm.attn_output");
    }

    launch_add(s.hidden, d_out, s.hidden, tokens * H, nullptr);
}
