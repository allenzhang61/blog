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

namespace {
// 统计 layer_index 之前同类型层的数量，得到本层在同类型序列中的下标。
size_t type_index_of(const TextConfig &config, size_t layer_index, const std::string &type) {
    size_t idx = 0;
    for (size_t i = 0; i < layer_index; ++i) {
        if (config.layer_types[i] == type) ++idx;
    }
    return idx;
}
} // namespace

DecoderLayer::DecoderLayer(const LayerWeights &weights, const TextConfig &text_config,
                           CudaWeightPool *pool, size_t layer_index)
    : text_config_(text_config),
      pool_(pool),
      layer_index_(layer_index),
      is_full_(weights.type == "full_attention"),
      input_norm_weight_(weights.attn_norm),
      post_norm_weight_(weights.ffn_norm),
      mlp_(weights.mlp, pool) {
    type_index_ = type_index_of(text_config, layer_index, weights.type);
    if (is_full_) {
        attn_ = std::make_unique<FullAttention>(weights.full, text_config, pool);
    } else {
        attn_ = std::make_unique<LinearAttention>(weights.lin, text_config, pool);
    }
}

void DecoderLayer::prefill(QwenSession &session, float *d_hidden, size_t input_size) {
    CudaScratch &scratch = session.scratch;
    const int hidden_size = text_config_.hidden_size;
    const size_t n = input_size * hidden_size;

    float *d_normed = scratch.ensure<float>(scratch_key::kTokenHiddenA, n, "layer normed");
    float *d_attn = scratch.ensure<float>(scratch_key::kMixerBuffer, n, "layer attn out");

    // h = x + attn( input_norm(x) )
    RMSNorm::forward(pool_, input_norm_weight_, d_hidden, d_normed, input_size, hidden_size,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->prefill(
            session, d_normed, d_attn, input_size, session.full_attn_kv_cache[type_index_]);
    } else {
        static_cast<LinearAttention *>(attn_.get())->prefill(
            session, d_normed, d_attn, input_size, session.linear_attn_recurrent_states[type_index_]);
    }
    launch_add(d_hidden, d_attn, d_hidden, static_cast<int>(n), nullptr);

    // y = h + mlp( post_norm(h) )
    float *d_post = scratch.ensure<float>(scratch_key::kTokenHiddenB, n, "layer post normed");
    float *d_mlp = scratch.ensure<float>(scratch_key::kMlpOut, n, "layer mlp out");
    RMSNorm::forward(pool_, post_norm_weight_, d_hidden, d_post, input_size, hidden_size,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(session, d_post, d_mlp, input_size, hidden_size);
    launch_add(d_hidden, d_mlp, d_hidden, static_cast<int>(n), nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer prefill 同步失败");
}

void DecoderLayer::decode(QwenSession &session, float *d_hidden, int pos) {
    CudaScratch &scratch = session.scratch;
    const int hidden = text_config_.hidden_size;

    float *d_normed = scratch.ensure<float>(scratch_key::kTokenHiddenA, hidden, "layer normed");
    float *d_attn = scratch.ensure<float>(scratch_key::kMixerBuffer, hidden, "layer attn out");

    RMSNorm::forward(pool_, input_norm_weight_, d_hidden, d_normed, 1, hidden,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->decode(
            session, d_normed, d_attn, pos, session.full_attn_kv_cache[type_index_]);
    } else {
        static_cast<LinearAttention *>(attn_.get())->decode(
            session, d_normed, d_attn, session.linear_attn_recurrent_states[type_index_]);
    }
    launch_add(d_hidden, d_attn, d_hidden, hidden, nullptr);

    float *d_post = scratch.ensure<float>(scratch_key::kTokenHiddenB, hidden, "layer post normed");
    float *d_mlp = scratch.ensure<float>(scratch_key::kMlpOut, hidden, "layer mlp out");
    RMSNorm::forward(pool_, post_norm_weight_, d_hidden, d_post, 1, hidden,
                     text_config_.rms_norm_eps, /*one_plus=*/true);
    mlp_.forward(session, d_post, d_mlp, 1, hidden);
    launch_add(d_hidden, d_mlp, d_hidden, hidden, nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer decode 同步失败");
}
