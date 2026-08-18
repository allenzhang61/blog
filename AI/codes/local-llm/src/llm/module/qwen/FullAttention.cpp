//
// Created by zhangyoulun on 9/8/2026.
//

#include "FullAttention.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

FullAttention::FullAttention(const FullAttnWeights &weights, const TextConfig &config,
                             CudaWeightPool *pool)
    : weights_(weights), config_(config), pool_(pool), type_index_(weights.type_index) {}

void FullAttention::prefill(QwenSession &session, const float *d_hidden, float *d_out, size_t input_size) {
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int hidden_size = config_.hidden_size;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;
    const int start_pos = kv.seq_len;

    // q_proj 输出 q_total*2（每 head 交错 [q, gate]）；k/v_proj 输出 kv_total。
    float *d_full_projection = scratch.ensure<float>(scratch_key::kFullProjection, input_size * static_cast<size_t>(q_total) * 2);
    float *d_full_k = scratch.ensure<float>(scratch_key::kFullK, input_size * static_cast<size_t>(kv_total));
    float *d_full_v = scratch.ensure<float>(scratch_key::kFullV, input_size * static_cast<size_t>(kv_total));
    float *d_full_q = scratch.ensure<float>(scratch_key::kFullQ, input_size * static_cast<size_t>(q_total));
    float *d_full_gate = scratch.ensure<float>(scratch_key::kFullGate, input_size * static_cast<size_t>(q_total));
    float *d_full_attn = scratch.ensure<float>(scratch_key::kFullAttn, input_size * static_cast<size_t>(q_total));

    // 输入激活转成权重 dtype（BF16/F16）后再投影。
    uint16_t *d_input_lowp =
        scratch.ensure<uint16_t>(scratch_key::kInputLowp, input_size * static_cast<size_t>(hidden_size));
    CudaWeight q_proj = pool_->cached_weight(weights_.q_proj)->try_dequant();
    GemmInput in = prepare_gemm_input(d_hidden, d_input_lowp, input_size * static_cast<size_t>(hidden_size), q_proj.type, nullptr);
    gemm_weight(pool_->handle, q_proj, in.ptr, d_full_projection, q_total * 2, hidden_size, input_size, in.type, "fullattn.q_proj");

    CudaWeight k_proj = pool_->cached_weight(weights_.k_proj)->try_dequant();
    gemm_weight(pool_->handle, k_proj, in.ptr, d_full_k, kv_total, hidden_size, input_size, in.type, "fullattn.k_proj");
    CudaWeight v_proj = pool_->cached_weight(weights_.v_proj)->try_dequant();
    gemm_weight(pool_->handle, v_proj, in.ptr, d_full_v, kv_total, hidden_size, input_size, in.type, "fullattn.v_proj");

    float *key_cache = static_cast<float *>(kv.key_cache.ptr);
    float *value_cache = static_cast<float *>(kv.value_cache.ptr);

    CudaWeight q_norm = pool_->cached_weight(weights_.q_norm)->try_dequant();
    launch_full_attention_q_batch(d_full_projection, static_cast<const uint16_t *>(q_norm.ptr), d_full_q, d_full_gate,
                                  input_size, n_heads, head_dim, start_pos, theta, partial, eps, nullptr);
    CudaWeight k_norm = pool_->cached_weight(weights_.k_norm)->try_dequant();
    launch_full_attention_kv_batch(d_full_k, d_full_v, static_cast<const uint16_t *>(k_norm.ptr),
                                   key_cache, value_cache, input_size, kv_heads, head_dim,
                                   /*max_seq_len=*/0, start_pos, theta, partial, eps, nullptr);
    launch_full_attention_attend_batch(d_full_q, d_full_gate, key_cache, value_cache, d_full_attn, input_size, n_heads,
                                       kv_heads, head_dim, /*max_seq_len=*/0, start_pos, nullptr);

    // attn 转成 o_proj 权重 dtype 后做输出投影。
    uint16_t *d_full_attn_lowp =
        scratch.ensure<uint16_t>(scratch_key::kFullAttnLowp, input_size * static_cast<size_t>(q_total));
    CudaWeight o_proj = pool_->cached_weight(weights_.o_proj)->try_dequant();
    GemmInput attn_in = prepare_gemm_input(d_full_attn, d_full_attn_lowp, input_size * static_cast<size_t>(q_total), o_proj.type, nullptr);
    // o_proj：[hidden, q_total] · attn[q_total, tokens] -> [hidden, tokens]。
    gemm_weight(pool_->handle, o_proj, attn_in.ptr, d_out, hidden_size, q_total, input_size, attn_in.type, "fullattn.o_proj");

    kv.seq_len += input_size;
    check_cuda(cudaDeviceSynchronize(), "FullAttention prefill 同步失败");
}

void FullAttention::decode(QwenSession &session, const float *d_hidden, float *d_out, int pos) {
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int hidden = config_.hidden_size;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;

    float *d_full_projection = scratch.ensure<float>(scratch_key::kFullProjection, static_cast<size_t>(q_total) * 2);
    float *d_full_k = scratch.ensure<float>(scratch_key::kFullK, kv_total);
    float *d_full_v = scratch.ensure<float>(scratch_key::kFullV, kv_total);
    float *d_full_q = scratch.ensure<float>(scratch_key::kFullQ, q_total);
    float *d_full_gate = scratch.ensure<float>(scratch_key::kFullGate, q_total);
    float *d_full_attn = scratch.ensure<float>(scratch_key::kFullAttn, q_total);

    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(scratch_key::kInputLowp, hidden);
    CudaWeight q_proj = pool_->cached_weight(weights_.q_proj)->try_dequant();
    GemmInput in = prepare_gemm_input(d_hidden, d_input_lowp, hidden, q_proj.type, nullptr);
    gemm_weight(pool_->handle, q_proj, in.ptr, d_full_projection, q_total * 2, hidden, 1, in.type, "fullattn.q_proj");

    CudaWeight k_proj = pool_->cached_weight(weights_.k_proj)->try_dequant();
    gemm_weight(pool_->handle, k_proj, in.ptr, d_full_k, kv_total, hidden, 1, in.type, "fullattn.k_proj");
    CudaWeight v_proj = pool_->cached_weight(weights_.v_proj)->try_dequant();
    gemm_weight(pool_->handle, v_proj, in.ptr, d_full_v, kv_total, hidden, 1, in.type, "fullattn.v_proj");

    float *key_cache = static_cast<float *>(kv.key_cache.ptr);
    float *value_cache = static_cast<float *>(kv.value_cache.ptr);

    CudaWeight q_norm = pool_->cached_weight(weights_.q_norm)->try_dequant();
    launch_full_attention_q(d_full_projection, static_cast<const uint16_t *>(q_norm.ptr), d_full_q, d_full_gate,
                            n_heads, head_dim, pos, theta, partial, eps, nullptr);
    CudaWeight k_norm = pool_->cached_weight(weights_.k_norm)->try_dequant();
    launch_full_attention_kv(d_full_k, d_full_v, static_cast<const uint16_t *>(k_norm.ptr),
                             key_cache, value_cache, kv_heads, head_dim, /*max_seq_len=*/0, pos,
                             theta, partial, eps, nullptr);
    launch_full_attention_attend(d_full_q, d_full_gate, key_cache, value_cache, d_full_attn, n_heads, kv_heads,
                                 head_dim, /*max_seq_len=*/0, pos, nullptr);

    uint16_t *d_full_attn_lowp = scratch.ensure<uint16_t>(scratch_key::kFullAttnLowp, q_total);
    CudaWeight o_proj = pool_->cached_weight(weights_.o_proj)->try_dequant();
    GemmInput attn_in = prepare_gemm_input(d_full_attn, d_full_attn_lowp, q_total, o_proj.type, nullptr);

    gemm_weight(pool_->handle, o_proj, attn_in.ptr, d_out, hidden, q_total, 1, attn_in.type, "fullattn.o_proj");

    kv.seq_len = pos + 1;
    check_cuda(cudaDeviceSynchronize(), "FullAttention decode 同步失败");
}
