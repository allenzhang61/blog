//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLA.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
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

    float *d_normed = scratch.ensure<float>(scratch_key::kNormed, static_cast<size_t>(input_size) * hidden_size);
    const std::vector<int64_t> act_shape = {static_cast<int64_t>(input_size),
                                            static_cast<int64_t>(hidden_size)};
    RMSNorm::forward(*lw_.attn_norm, hidden,
                     Tensor::gpu_activation(d_normed, act_shape),
                     config_.rms_norm_eps, /*one_plus=*/false);

    float *d_q = scratch.ensure<float>(scratch_key::kQ, static_cast<size_t>(input_size) * q_dim);
    {
        CudaWeight attn_q = lw_.attn_q->cached_weight()->try_dequant();
        uint16_t *d_normed_lowp = scratch.ensure<uint16_t>(scratch_key::kNormedLowp, static_cast<size_t>(input_size) * hidden_size);
        GemmInput q_in = prepare_gemm_input(d_normed, d_normed_lowp, input_size * hidden_size, attn_q.type, nullptr);
        gemm_weight(global_cuda_weight_pool().handle, attn_q, q_in.ptr, d_q, q_dim, hidden_size, input_size, q_in.type, "ds.gemm.attn_q");
    }
    if (input_size == 1) {
        launch_mla_rope_q(d_q, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    } else {
        launch_mla_rope_q_batch(d_q, input_size, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    }

    float *d_kv_a = scratch.ensure<float>(scratch_key::kKvA, static_cast<size_t>(input_size) * kv_total);
    {
        CudaWeight attn_kv_a_mqa = lw_.attn_kv_a_mqa->cached_weight()->try_dequant();
        uint16_t *d_normed_lowp = scratch.ensure<uint16_t>(scratch_key::kNormedLowp, static_cast<size_t>(input_size) * hidden_size);
        GemmInput kv_a_in = prepare_gemm_input(d_normed, d_normed_lowp, input_size * hidden_size, attn_kv_a_mqa.type, nullptr);
        gemm_weight(global_cuda_weight_pool().handle, attn_kv_a_mqa, kv_a_in.ptr, d_kv_a, kv_total, hidden_size, input_size, kv_a_in.type, "ds.gemm.kv_a");
    }

    float *d_cache = static_cast<float *>(session.kv_caches[layer].cache.ptr);
    CudaWeight *kv_a_norm_weight = lw_.attn_kv_a_norm->cached_weight();
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
    float *d_kv_b_out = scratch.ensure<float>(scratch_key::kKvBOut, static_cast<size_t>(seq) * kvb_out);
    {
        CudaWeight attn_kv_b = lw_.attn_kv_b->cached_weight()->try_dequant();
        uint16_t *d_latent_lowp = scratch.ensure<uint16_t>(scratch_key::kLatentLowp, static_cast<size_t>(seq) * kv_lora);
        float *latent_f32 = scratch.ensure<float>(scratch_key::kAttn, static_cast<size_t>(seq) * kv_lora);
        cuda_memcpy2d_d2d(latent_f32, kv_lora * sizeof(float), d_cache,
                          kv_total * sizeof(float), kv_lora * sizeof(float), seq,
                          "ds.gather.latent");
        GemmInput latent_in = prepare_gemm_input(latent_f32, d_latent_lowp, seq * kv_lora, attn_kv_b.type, nullptr);
        gemm_weight(global_cuda_weight_pool().handle, attn_kv_b, latent_in.ptr, d_kv_b_out, kvb_out, kv_lora, seq, latent_in.type, "ds.gemm.kv_b");
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
        CudaWeight attn_output = lw_.attn_output->cached_weight()->try_dequant();
        const int in_dim = n_heads * v_head; // 2048
        uint16_t *d_attn_lowp = scratch.ensure<uint16_t>(scratch_key::kAttnLowp, static_cast<size_t>(input_size) * in_dim);
        GemmInput attn_in = prepare_gemm_input(d_attn, d_attn_lowp, input_size * in_dim, attn_output.type, nullptr);
        gemm_weight(global_cuda_weight_pool().handle, attn_output, attn_in.ptr, d_attn_out, hidden_size, in_dim, input_size, attn_in.type, "ds.gemm.attn_output");
    }

    launch_add(d_hidden, d_attn_out, d_hidden, input_size * hidden_size, nullptr);
}
