//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLA.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekTrace.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <cstdint>

MLA::MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {
}

void MLA::forward(DeepseekSession &session, const GPUTensor &g_hidden_f32, int start_pos) {
    const int64_t input_size = g_hidden_f32.rows();
    auto &scratch = session.cuda_scratch;
    const int layer = lw_.layer_index;
    const int64_t hidden_size = config_.hidden_size;
    const int n_heads = config_.num_heads;
    const int qk_nope = config_.qk_nope_head_dim;
    const int qk_rope = config_.qk_rope_head_dim;
    const int qk_head = qk_nope + qk_rope; //config_.qk_head_dim(); // 192
    const int v_head = config_.v_head_dim; // 128
    const int kv_lora = config_.kv_lora_rank; // 512
    const int kv_total = kv_lora + qk_rope; // 576
    const int kvb_out = n_heads * (qk_nope + v_head); // 4096
    const GPUTensor &g_inv_freq_f32 = session.g_inv_freq_f32;

    auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {input_size, hidden_size}, DType::F32);
    RMSNorm::forward(*lw_.s_attn_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    deepseek_trace::tensor(session, g_normed_f32, "attn_normed", session.trace_pos, session.trace_layer);

    auto g_q_f32 = GPUTensor(scratch, scratch_key::kQ,
                                  {
                                      input_size,
                                      static_cast<int64_t>(n_heads),
                                      static_cast<int64_t>(qk_nope + qk_rope) //(qk_head)
                                  }, DType::F32);
    TensorTool::gemm(*lw_.s_attn_q, g_normed_f32, g_q_f32, scratch, scratch_key::kNormedLowp, "ds.gemm.d_attn_q");
    TensorTool::mla_rope_q(g_q_f32, input_size, n_heads, qk_nope, qk_rope, start_pos, g_inv_freq_f32);
    deepseek_trace::tensor(session, g_q_f32, "attn_q_rope", session.trace_pos, session.trace_layer);

    GPUTensor g_kv_a_f32 = GPUTensor(
        scratch, scratch_key::kKvA,
        {input_size, static_cast<int64_t>(kv_total)}, DType::F32);
    TensorTool::gemm(*lw_.s_attn_kv_a_mqa, g_normed_f32, g_kv_a_f32, scratch, scratch_key::kNormedLowp, "ds.gemm.kv_a");
    deepseek_trace::tensor(session, g_kv_a_f32, "kv_a", session.trace_pos, session.trace_layer);

    GPUTensor &g_kv_cache_f32 = session.kv_caches[layer].g_cache_f32;
    TensorTool::mla_kv_a(g_kv_a_f32, *lw_.s_attn_kv_a_norm, g_kv_cache_f32, input_size, kv_lora, qk_rope,
                         start_pos, g_inv_freq_f32, config_.rms_norm_eps);
    session.kv_caches[layer].seq_len = start_pos + static_cast<int>(input_size);

    const int64_t seq = start_pos + input_size;
    GPUTensor g_kv_b_out_f32 = GPUTensor(scratch, scratch_key::kKvBOut,
                                         {
                                             seq,
                                             static_cast<int64_t>(n_heads * (qk_nope + v_head)) //kvb_out
                                         },
                                         DType::F32);
    {
        GPUTensor g_latent_f32 = GPUTensor(
            scratch, scratch_key::kAttn,
            {seq, static_cast<int64_t>(kv_lora)}, DType::F32);
        cuda_memcpy2d_d2d(g_latent_f32.data(), kv_lora * sizeof(float), g_kv_cache_f32.data(),
                          kv_total * sizeof(float), kv_lora * sizeof(float), static_cast<size_t>(seq),
                          "ds.gather.latent");
        deepseek_trace::tensor(session, g_latent_f32, "kv_latent", session.trace_pos, session.trace_layer);
        TensorTool::gemm(*lw_.s_attn_kv_b, g_latent_f32, g_kv_b_out_f32, scratch, scratch_key::kLatentLowp,
                         "ds.gemm.kv_b");
        deepseek_trace::tensor(session, g_kv_b_out_f32, "kv_b_out", session.trace_pos, session.trace_layer);
    }

    GPUTensor g_attn_f32 = GPUTensor(
        scratch, scratch_key::kAttn,
        {input_size, static_cast<int64_t>(n_heads * v_head)}, DType::F32);
    TensorTool::mla_attend(g_q_f32, g_kv_b_out_f32, g_kv_cache_f32, g_attn_f32, input_size, n_heads, qk_nope, qk_rope,
                           v_head, kv_lora, start_pos, session.attn_softmax_scale);
    deepseek_trace::tensor(session, g_attn_f32, "attn_ctx", session.trace_pos, session.trace_layer);

    GPUTensor g_attn_out_f32 = GPUTensor(scratch, scratch_key::kAttnOut, {input_size, hidden_size}, DType::F32);
    TensorTool::gemm(*lw_.s_attn_output, g_attn_f32, g_attn_out_f32, scratch, scratch_key::kAttnLowp,
                     "ds.gemm.d_attn_output");
    deepseek_trace::tensor(session, g_attn_out_f32, "attn_out", session.trace_pos, session.trace_layer);
    TensorTool::add(g_hidden_f32, g_attn_out_f32, g_hidden_f32);
}
