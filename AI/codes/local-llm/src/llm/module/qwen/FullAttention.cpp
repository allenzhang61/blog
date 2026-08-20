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
#include "backend/cuda/ops/kernel.cuh"

FullAttention::FullAttention(const FullAttnWeights &weights, const TextConfig &config)
    : weights_(weights), config_(config), type_index_(weights.type_index) {}

void FullAttention::prefill(QwenSession &session, const Tensor &hidden, const Tensor &out) {
    const size_t input_size = static_cast<size_t>(hidden.rows());
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
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

    Tensor full_projection = Tensor::gpu_view(d_full_projection, {static_cast<int64_t>(input_size), static_cast<int64_t>(q_total * 2)});
    Tensor full_k = Tensor::gpu_view(d_full_k, {static_cast<int64_t>(input_size), static_cast<int64_t>(kv_total)});
    Tensor full_v = Tensor::gpu_view(d_full_v, {static_cast<int64_t>(input_size), static_cast<int64_t>(kv_total)});
    weights_.q_proj.to_gpu();
    weights_.q_proj.gemm(hidden, full_projection, scratch, scratch_key::kInputLowp, "fullattn.q_proj");
    weights_.k_proj.to_gpu();
    weights_.k_proj.gemm(hidden, full_k, scratch, scratch_key::kInputLowp, "fullattn.k_proj");
    weights_.v_proj.to_gpu();
    weights_.v_proj.gemm(hidden, full_v, scratch, scratch_key::kInputLowp, "fullattn.v_proj");

    float *key_cache = static_cast<float *>(kv.key_cache.ptr);
    float *value_cache = static_cast<float *>(kv.value_cache.ptr);

    weights_.q_norm.to_gpu();
    launch_full_attention_q_batch(d_full_projection, static_cast<const uint16_t *>(weights_.q_norm.weight_gpu_data()), d_full_q, d_full_gate,
                                  input_size, n_heads, head_dim, start_pos, theta, partial, eps, nullptr);
    weights_.k_norm.to_gpu();
    launch_full_attention_kv_batch(d_full_k, d_full_v, static_cast<const uint16_t *>(weights_.k_norm.weight_gpu_data()),
                                   key_cache, value_cache, input_size, kv_heads, head_dim,
                                   /*max_seq_len=*/0, start_pos, theta, partial, eps, nullptr);
    launch_full_attention_attend_batch(d_full_q, d_full_gate, key_cache, value_cache, d_full_attn, input_size, n_heads,
                                       kv_heads, head_dim, /*max_seq_len=*/0, start_pos, nullptr);

    // o_proj：[hidden, q_total] · attn[q_total, tokens] -> [hidden, tokens]。
    Tensor full_attn = Tensor::gpu_view(d_full_attn, {static_cast<int64_t>(input_size), static_cast<int64_t>(q_total)});
    weights_.o_proj.to_gpu();
    weights_.o_proj.gemm(full_attn, out, scratch, scratch_key::kFullAttnLowp, "fullattn.o_proj");

    kv.seq_len += input_size;
    check_cuda(cudaDeviceSynchronize(), "FullAttention prefill 同步失败");
}

void FullAttention::decode(QwenSession &session, const Tensor &hidden, const Tensor &out, int pos) {
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
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

    Tensor full_projection = Tensor::gpu_view(d_full_projection, {1, static_cast<int64_t>(q_total * 2)});
    Tensor full_k = Tensor::gpu_view(d_full_k, {1, static_cast<int64_t>(kv_total)});
    Tensor full_v = Tensor::gpu_view(d_full_v, {1, static_cast<int64_t>(kv_total)});
    weights_.q_proj.to_gpu();
    weights_.q_proj.gemm(hidden, full_projection, scratch, scratch_key::kInputLowp, "fullattn.q_proj");
    weights_.k_proj.to_gpu();
    weights_.k_proj.gemm(hidden, full_k, scratch, scratch_key::kInputLowp, "fullattn.k_proj");
    weights_.v_proj.to_gpu();
    weights_.v_proj.gemm(hidden, full_v, scratch, scratch_key::kInputLowp, "fullattn.v_proj");

    float *key_cache = static_cast<float *>(kv.key_cache.ptr);
    float *value_cache = static_cast<float *>(kv.value_cache.ptr);

    weights_.q_norm.to_gpu();
    launch_full_attention_q(d_full_projection, static_cast<const uint16_t *>(weights_.q_norm.weight_gpu_data()), d_full_q, d_full_gate,
                            n_heads, head_dim, pos, theta, partial, eps, nullptr);
    weights_.k_norm.to_gpu();
    launch_full_attention_kv(d_full_k, d_full_v, static_cast<const uint16_t *>(weights_.k_norm.weight_gpu_data()),
                             key_cache, value_cache, kv_heads, head_dim, /*max_seq_len=*/0, pos,
                             theta, partial, eps, nullptr);
    launch_full_attention_attend(d_full_q, d_full_gate, key_cache, value_cache, d_full_attn, n_heads, kv_heads,
                                 head_dim, /*max_seq_len=*/0, pos, nullptr);

    Tensor full_attn = Tensor::gpu_view(d_full_attn, {1, static_cast<int64_t>(q_total)});
    weights_.o_proj.to_gpu();
    weights_.o_proj.gemm(full_attn, out, scratch, scratch_key::kFullAttnLowp, "fullattn.o_proj");

    kv.seq_len = pos + 1;
    check_cuda(cudaDeviceSynchronize(), "FullAttention decode 同步失败");
}
