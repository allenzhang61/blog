//
// Created by zhangyoulun on 9/8/2026.
//

#include "DecoderLayer.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "FullAttention.h"
#include "LinearAttention.h"
#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"

DecoderLayer::DecoderLayer(const LayerWeights &weights, const TextConfig &config,
                           CudaWeightPool *pool)
    : text_config_(config),
      pool_(pool),
      is_full_(weights.type == "full_attention"),
      input_norm_weight_(weights.input_layernorm),
      post_norm_weight_(weights.post_attention_layernorm),
      mlp_(weights.mlp, pool) {
    if (is_full_) {
        attn_ = std::make_unique<FullAttention>(weights.full, config, pool);
    } else {
        attn_ = std::make_unique<LinearAttention>(weights.lin, config, pool);
    }
}

void DecoderLayer::prefill(QwenSession &session, float *d_hidden, size_t input_size) {
    CudaScratch &scratch = session.scratch;
    const int hidden_size = text_config_.hidden_size;
    const size_t n = input_size * hidden_size;

    float *d_token_hidden_a = scratch.ensure<float>(scratch_key::kTokenHiddenA, n);
    float *d_mixer = scratch.ensure<float>(scratch_key::kMixer, n);

    // h = x + attn( input_norm(x) )
    RMSNorm::forward(pool_, input_norm_weight_, d_hidden, d_token_hidden_a, input_size, hidden_size,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->prefill(
            session, d_token_hidden_a, d_mixer, input_size);
    } else {
        static_cast<LinearAttention *>(attn_.get())->prefill(
            session, d_token_hidden_a, d_mixer, input_size);
    }
    launch_add(d_hidden, d_mixer, d_hidden, static_cast<int>(n), nullptr);

    // y = h + mlp( post_norm(h) )
    float *d_token_hidden_b = scratch.ensure<float>(scratch_key::kTokenHiddenB, n);
    float *d_mlp_out = scratch.ensure<float>(scratch_key::kMlpOut, n);
    RMSNorm::forward(pool_, post_norm_weight_, d_hidden, d_token_hidden_b, input_size, hidden_size,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(session, d_token_hidden_b, d_mlp_out, input_size, hidden_size);
    launch_add(d_hidden, d_mlp_out, d_hidden, static_cast<int>(n), nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer prefill 同步失败");
}

void DecoderLayer::decode(QwenSession &session, float *d_hidden, int pos) {
    CudaScratch &scratch = session.scratch;
    const int hidden = text_config_.hidden_size;

    float *d_token_hidden_a = scratch.ensure<float>(scratch_key::kTokenHiddenA, hidden);
    float *d_mixer = scratch.ensure<float>(scratch_key::kMixer, hidden);

    RMSNorm::forward(pool_, input_norm_weight_, d_hidden, d_token_hidden_a, 1, hidden,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->decode(
            session, d_token_hidden_a, d_mixer, pos);
    } else {
        static_cast<LinearAttention *>(attn_.get())->decode(
            session, d_token_hidden_a, d_mixer);
    }
    launch_add(d_hidden, d_mixer, d_hidden, hidden, nullptr);

    float *d_token_hidden_b = scratch.ensure<float>(scratch_key::kTokenHiddenB, hidden);
    float *d_mlp_out = scratch.ensure<float>(scratch_key::kMlpOut, hidden);
    RMSNorm::forward(pool_, post_norm_weight_, d_hidden, d_token_hidden_b, 1, hidden,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(session, d_token_hidden_b, d_mlp_out, 1, hidden);
    launch_add(d_hidden, d_mlp_out, d_hidden, hidden, nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer decode 同步失败");
}
