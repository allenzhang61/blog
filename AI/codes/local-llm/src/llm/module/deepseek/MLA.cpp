//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLA.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cuda_runtime.h>

MLA::MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {
}

void MLA::forward(DeepseekSession &session, const GPUTensor &g_hidden, int start_pos) {
    const int input_size = static_cast<int>(g_hidden.rows());
    auto &scratch = session.scratch;
    const int layer = lw_.layer_index;
    const int hidden_size = config_.hidden_size;
    const int n_heads = config_.num_heads;
    const int qk_nope = config_.qk_nope_head_dim;
    const int qk_rope = config_.qk_rope_head_dim;
    const int qk_head = config_.qk_head_dim(); // 192
    const int v_head = config_.v_head_dim; // 128
    const int kv_lora = config_.kv_lora_rank; // 512
    const int kv_total = kv_lora + qk_rope; // 576
    const int q_dim = n_heads * qk_head; // 3072
    const int kvb_out = n_heads * (qk_nope + v_head); // 4096
    const GPUTensor &g_inv_freq = session.g_inv_freq;

    GPUTensor g_normed = GPUTensor(scratch, scratch_key::kNormed, {
                                       static_cast<int64_t>(input_size),
                                       static_cast<int64_t>(hidden_size)
                                   }, DType::F32);
    RMSNorm::forward(*lw_.s_attn_norm, g_hidden, g_normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    GPUTensor g_q = GPUTensor(scratch, scratch_key::kQ,
                              {static_cast<int64_t>(input_size), static_cast<int64_t>(q_dim)}, DType::F32);
    TensorTool::gemm(*lw_.s_attn_q, g_normed, g_q, scratch, scratch_key::kNormedLowp, "ds.gemm.d_attn_q");
    if (input_size == 1) {
        TensorTool::mla_rope_q(g_q, n_heads, qk_nope, qk_rope, start_pos, g_inv_freq);
    } else {
        TensorTool::mla_rope_q_batch(g_q, n_heads, qk_nope, qk_rope, start_pos, g_inv_freq);
    }

    GPUTensor g_kv_a = GPUTensor(
        scratch, scratch_key::kKvA,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(kv_total)}, DType::F32);
    TensorTool::gemm(*lw_.s_attn_kv_a_mqa, g_normed, g_kv_a, scratch, scratch_key::kNormedLowp, "ds.gemm.kv_a");

    GPUTensor &g_kv_cache = session.kv_caches[layer].g_cache;
    if (input_size == 1) {
        TensorTool::mla_kv_a(g_kv_a, *lw_.s_attn_kv_a_norm, g_kv_cache, kv_lora, qk_rope, session.max_seq_len,
                             start_pos, g_inv_freq, config_.rms_norm_eps);
    } else {
        TensorTool::mla_kv_a_batch(g_kv_a, *lw_.s_attn_kv_a_norm, g_kv_cache, kv_lora, qk_rope,
                                   session.max_seq_len, start_pos, g_inv_freq, config_.rms_norm_eps);
    }
    session.kv_caches[layer].seq_len = start_pos + input_size;

    const int seq = start_pos + input_size;
    GPUTensor g_kv_b_out = GPUTensor(
        scratch, scratch_key::kKvBOut,
        {static_cast<int64_t>(seq), static_cast<int64_t>(kvb_out)}, DType::F32);
    {
        GPUTensor g_latent = GPUTensor(
            scratch, scratch_key::kAttn,
            {static_cast<int64_t>(seq), static_cast<int64_t>(kv_lora)}, DType::F32);
        cuda_memcpy2d_d2d(g_latent.data(), kv_lora * sizeof(float), g_kv_cache.data(),
                          kv_total * sizeof(float), kv_lora * sizeof(float), seq,
                          "ds.gather.latent");
        TensorTool::gemm(*lw_.s_attn_kv_b, g_latent, g_kv_b_out, scratch, scratch_key::kLatentLowp, "ds.gemm.kv_b");
    }

    GPUTensor g_attn = GPUTensor(
        scratch, scratch_key::kAttn,
        {static_cast<int64_t>(input_size), static_cast<int64_t>(n_heads * v_head)}, DType::F32);
    if (input_size == 1) {
        TensorTool::mla_attend(g_q, g_kv_b_out, g_kv_cache, g_attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                               session.max_seq_len, start_pos, session.attn_softmax_scale);
    } else {
        TensorTool::mla_attend_batch(g_q, g_kv_b_out, g_kv_cache, g_attn, n_heads, qk_nope, qk_rope,
                                     v_head, kv_lora, session.max_seq_len, start_pos,
                                     session.attn_softmax_scale);
    }

    GPUTensor g_attn_out = GPUTensor(scratch, scratch_key::kAttnOut, {
                                         static_cast<int64_t>(input_size),
                                         static_cast<int64_t>(hidden_size)
                                     }, DType::F32);
    TensorTool::gemm(*lw_.s_attn_output, g_attn, g_attn_out, scratch, scratch_key::kAttnLowp, "ds.gemm.d_attn_output");
    TensorTool::add(g_hidden, g_attn_out, g_hidden);
}
