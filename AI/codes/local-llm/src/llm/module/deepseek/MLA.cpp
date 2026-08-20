//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLA.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"

#include <cuda_runtime.h>

MLA::MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {}

void MLA::forward(DeepseekSession &session, const Tensor &hidden, int start_pos) {
    float *d_hidden = hidden.gpu_f32();
    const int input_size = static_cast<int>(hidden.rows());
    auto &scratch = session.scratch;
    const int layer = lw_.layer_index;
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

    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};
    Tensor normed = Tensor::gpu_scratch(scratch, scratch_key::kNormed, act_shape);
    RMSNorm::forward(*lw_.attn_norm, hidden, normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    float *d_q = scratch.ensure<float>(scratch_key::kQ, static_cast<size_t>(input_size) * q_dim);
    Tensor q = Tensor::gpu_view(d_q, {static_cast<int64_t>(input_size), static_cast<int64_t>(q_dim)});
    lw_.attn_q->to_gpu();
    lw_.attn_q->gemm(normed, q, scratch, scratch_key::kNormedLowp, "ds.gemm.attn_q");
    if (input_size == 1) {
        launch_mla_rope_q(d_q, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    } else {
        launch_mla_rope_q_batch(d_q, input_size, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    }

    float *d_kv_a = scratch.ensure<float>(scratch_key::kKvA, static_cast<size_t>(input_size) * kv_total);
    Tensor kv_a = Tensor::gpu_view(d_kv_a, {static_cast<int64_t>(input_size), static_cast<int64_t>(kv_total)});
    lw_.attn_kv_a_mqa->to_gpu();
    lw_.attn_kv_a_mqa->gemm(normed, kv_a, scratch, scratch_key::kNormedLowp, "ds.gemm.kv_a");

    float *d_cache = static_cast<float *>(session.kv_caches[layer].cache.ptr);
    lw_.attn_kv_a_norm->to_gpu();
    const float *kv_a_norm = static_cast<const float *>(lw_.attn_kv_a_norm->weight_gpu_data());
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
    float *d_kv_b_out = scratch.ensure<float>(scratch_key::kKvBOut, static_cast<size_t>(seq) * kvb_out);
    {
        float *latent_f32 = scratch.ensure<float>(scratch_key::kAttn, static_cast<size_t>(seq) * kv_lora);
        cuda_memcpy2d_d2d(latent_f32, kv_lora * sizeof(float), d_cache,
                          kv_total * sizeof(float), kv_lora * sizeof(float), seq,
                          "ds.gather.latent");
        Tensor latent = Tensor::gpu_view(latent_f32, {static_cast<int64_t>(seq), static_cast<int64_t>(kv_lora)});
        Tensor kv_b_out = Tensor::gpu_view(d_kv_b_out, {static_cast<int64_t>(seq), static_cast<int64_t>(kvb_out)});
        lw_.attn_kv_b->to_gpu();
        lw_.attn_kv_b->gemm(latent, kv_b_out, scratch, scratch_key::kLatentLowp, "ds.gemm.kv_b");
    }

    float *d_attn = scratch.ensure<float>(scratch_key::kAttn, static_cast<size_t>(input_size) * n_heads * v_head);
    if (input_size == 1) {
        launch_mla_attend(d_q, d_kv_b_out, d_cache, d_attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                          session.max_seq_len, start_pos, session.attn_softmax_scale, nullptr);
    } else {
        launch_mla_attend_batch(d_q, d_kv_b_out, d_cache, d_attn, input_size, n_heads, qk_nope, qk_rope,
                                v_head, kv_lora, session.max_seq_len, start_pos,
                                session.attn_softmax_scale, nullptr);
    }

    float *d_attn_out = scratch.ensure<float>(scratch_key::kAttnOut, static_cast<size_t>(input_size) * hidden_size);
    {
        const int in_dim = n_heads * v_head; // 2048
        Tensor attn = Tensor::gpu_view(d_attn, {static_cast<int64_t>(input_size), static_cast<int64_t>(in_dim)});
        Tensor attn_out = Tensor::gpu_view(d_attn_out, act_shape);
        lw_.attn_output->to_gpu();
        lw_.attn_output->gemm(attn, attn_out, scratch, scratch_key::kAttnLowp, "ds.gemm.attn_output");
    }

    launch_add(d_hidden, d_attn_out, d_hidden, input_size * hidden_size, nullptr);
}
