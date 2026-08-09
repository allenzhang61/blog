//
// Created by zhangyoulun on 9/8/2026.
//

#include "FullAttention.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "llm/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

FullAttention::FullAttention(const FullAttnWeights &weights, const TextConfig &config,
                             CudaWeightPool *pool)
    : weights_(weights), config_(config), pool_(pool) {}

void FullAttention::prefill(const float *d_hidden, float *d_out, int tokens,
                            FullAttnKVCache &kv, QwenForwardScratch &scratch) {
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int hidden = config_.hidden_size;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;
    const int start_pos = kv.seq_len;

    CudaWeight *q_proj = pool_->cached_weight(weights_.q_proj);
    CudaWeight *k_proj = pool_->cached_weight(weights_.k_proj);
    CudaWeight *v_proj = pool_->cached_weight(weights_.v_proj);
    CudaWeight *q_norm = pool_->cached_weight(weights_.q_norm);
    CudaWeight *k_norm = pool_->cached_weight(weights_.k_norm);
    CudaWeight *o_proj = pool_->cached_weight(weights_.o_proj);
    if (!q_proj || !k_proj || !v_proj || !q_norm || !k_norm || !o_proj) {
        throw std::runtime_error("FullAttention 权重上传失败");
    }

    // q_proj 输出 q_total*2（每 head 交错 [q, gate]）；k/v_proj 输出 kv_total。
    float *d_qgate = scratch.full_projection.ensure(static_cast<size_t>(tokens) * q_total * 2, "full qgate");
    float *d_k = scratch.full_k.ensure(static_cast<size_t>(tokens) * kv_total, "full k");
    float *d_v = scratch.full_v.ensure(static_cast<size_t>(tokens) * kv_total, "full v");
    float *d_q = scratch.full_q.ensure(static_cast<size_t>(tokens) * q_total, "full q");
    float *d_g = scratch.full_gate.ensure(static_cast<size_t>(tokens) * q_total, "full gate");
    float *d_attn = scratch.full_attn.ensure(static_cast<size_t>(tokens) * q_total, "full attn");

    gemm_weight(pool_->handle, *q_proj, q_total * 2, hidden, d_hidden, CUDA_R_32F, tokens, d_qgate);
    gemm_weight(pool_->handle, *k_proj, kv_total, hidden, d_hidden, CUDA_R_32F, tokens, d_k);
    gemm_weight(pool_->handle, *v_proj, kv_total, hidden, d_hidden, CUDA_R_32F, tokens, d_v);

    float *key_cache = static_cast<float *>(kv.key_cache.ptr);
    float *value_cache = static_cast<float *>(kv.value_cache.ptr);

    launch_full_attention_q_batch(d_qgate, static_cast<const uint16_t *>(q_norm->ptr), d_q, d_g,
                                  tokens, n_heads, head_dim, start_pos, theta, partial, eps, nullptr);
    launch_full_attention_kv_batch(d_k, d_v, static_cast<const uint16_t *>(k_norm->ptr),
                                   key_cache, value_cache, tokens, kv_heads, head_dim,
                                   /*max_seq_len=*/0, start_pos, theta, partial, eps, nullptr);
    launch_full_attention_attend_batch(d_q, d_g, key_cache, value_cache, d_attn, tokens, n_heads,
                                       kv_heads, head_dim, /*max_seq_len=*/0, start_pos, nullptr);

    // o_proj：[hidden, q_total] · attn[q_total, tokens] -> [hidden, tokens]。
    gemm_weight(pool_->handle, *o_proj, hidden, q_total, d_attn, CUDA_R_32F, tokens, d_out);

    kv.seq_len += tokens;
    check_cuda(cudaDeviceSynchronize(), "FullAttention prefill 同步失败");
}

void FullAttention::decode(const float *d_hidden, float *d_out, int pos,
                           FullAttnKVCache &kv, QwenForwardScratch &scratch) {
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int hidden = config_.hidden_size;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;

    CudaWeight *q_proj = pool_->cached_weight(weights_.q_proj);
    CudaWeight *k_proj = pool_->cached_weight(weights_.k_proj);
    CudaWeight *v_proj = pool_->cached_weight(weights_.v_proj);
    CudaWeight *q_norm = pool_->cached_weight(weights_.q_norm);
    CudaWeight *k_norm = pool_->cached_weight(weights_.k_norm);
    CudaWeight *o_proj = pool_->cached_weight(weights_.o_proj);
    if (!q_proj || !k_proj || !v_proj || !q_norm || !k_norm || !o_proj) {
        throw std::runtime_error("FullAttention 权重上传失败");
    }

    float *d_qgate = scratch.full_projection.ensure(static_cast<size_t>(q_total) * 2, "full qgate");
    float *d_k = scratch.full_k.ensure(kv_total, "full k");
    float *d_v = scratch.full_v.ensure(kv_total, "full v");
    float *d_q = scratch.full_q.ensure(q_total, "full q");
    float *d_g = scratch.full_gate.ensure(q_total, "full gate");
    float *d_attn = scratch.full_attn.ensure(q_total, "full attn");

    gemm_weight(pool_->handle, *q_proj, q_total * 2, hidden, d_hidden, CUDA_R_32F, 1, d_qgate);
    gemm_weight(pool_->handle, *k_proj, kv_total, hidden, d_hidden, CUDA_R_32F, 1, d_k);
    gemm_weight(pool_->handle, *v_proj, kv_total, hidden, d_hidden, CUDA_R_32F, 1, d_v);

    float *key_cache = static_cast<float *>(kv.key_cache.ptr);
    float *value_cache = static_cast<float *>(kv.value_cache.ptr);

    launch_full_attention_q(d_qgate, static_cast<const uint16_t *>(q_norm->ptr), d_q, d_g,
                            n_heads, head_dim, pos, theta, partial, eps, nullptr);
    launch_full_attention_kv(d_k, d_v, static_cast<const uint16_t *>(k_norm->ptr),
                             key_cache, value_cache, kv_heads, head_dim, /*max_seq_len=*/0, pos,
                             theta, partial, eps, nullptr);
    launch_full_attention_attend(d_q, d_g, key_cache, value_cache, d_attn, n_heads, kv_heads,
                                 head_dim, /*max_seq_len=*/0, pos, nullptr);

    gemm_weight(pool_->handle, *o_proj, hidden, q_total, d_attn, CUDA_R_32F, 1, d_out);

    kv.seq_len = pos + 1;
    check_cuda(cudaDeviceSynchronize(), "FullAttention decode 同步失败");
}
