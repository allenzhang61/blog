//
// Created by zhangyoulun on 9/8/2026.
//

#include "DecoderLayer.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "FullAttention.h"
#include "LinearAttention.h"
#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenSession.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"
#include "backend/cuda/ops/kernel.cuh"

namespace {
// 统计 layer_index 之前同类型层的数量，得到本层在同类型序列中的下标。
int type_index_of(const TextConfig &config, int layer_index, const std::string &type) {
    int idx = 0;
    for (int i = 0; i < layer_index; ++i) {
        if (config.layer_types[i] == type) ++idx;
    }
    return idx;
}
} // namespace

DecoderLayer::DecoderLayer(const LayerWeights &weights, const TextConfig &text_config,
                           CudaWeightPool *pool, int layer_index)
    : text_config_(text_config),
      layer_index_(layer_index),
      is_full_(weights.type == "full_attention"),
      input_norm_(weights.input_norm, pool, text_config.rms_norm_eps),
      post_norm_(weights.post_norm, pool, text_config.rms_norm_eps),
      mlp_(weights.mlp, pool) {
    type_index_ = type_index_of(text_config, layer_index, weights.type);
    if (is_full_) {
        attn_ = std::make_unique<FullAttention>(weights.full, text_config, pool);
    } else {
        attn_ = std::make_unique<LinearAttention>(weights.lin, text_config, pool);
    }
}

void DecoderLayer::prefill(float *d_hidden, int tokens, QwenSession &session,
                           QwenForwardScratch &scratch) {
    const int hidden = text_config_.hidden_size;
    const size_t n = static_cast<size_t>(tokens) * hidden;

    float *d_normed = scratch.token_hidden_a.ensure(n, "layer normed");
    float *d_attn = scratch.mixer_buffer.ensure(n, "layer attn out");

    // h = x + attn( input_norm(x) )
    input_norm_.forward(d_hidden, d_normed, tokens, hidden);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->prefill(
            d_normed, d_attn, tokens, session.fullAttnKVCaches[type_index_], scratch);
    } else {
        static_cast<LinearAttention *>(attn_.get())->prefill(
            d_normed, d_attn, tokens, session.linearAttnRecurrentStates[type_index_], scratch);
    }
    launch_add(d_hidden, d_attn, d_hidden, static_cast<int>(n), nullptr);

    // y = h + mlp( post_norm(h) )
    float *d_post = scratch.token_hidden_b.ensure(n, "layer post normed");
    float *d_mlp = scratch.mlp_out_buffer.ensure(n, "layer mlp out");
    post_norm_.forward(d_hidden, d_post, tokens, hidden);
    mlp_.forward(d_post, d_mlp, tokens, hidden, scratch);
    launch_add(d_hidden, d_mlp, d_hidden, static_cast<int>(n), nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer prefill 同步失败");
}

void DecoderLayer::decode(float *d_hidden, int pos, QwenSession &session,
                          QwenForwardScratch &scratch) {
    const int hidden = text_config_.hidden_size;

    float *d_normed = scratch.token_hidden_a.ensure(hidden, "layer normed");
    float *d_attn = scratch.mixer_buffer.ensure(hidden, "layer attn out");

    input_norm_.forward(d_hidden, d_normed, 1, hidden);
    if (is_full_) {
        static_cast<FullAttention *>(attn_.get())->decode(
            d_normed, d_attn, pos, session.fullAttnKVCaches[type_index_], scratch);
    } else {
        static_cast<LinearAttention *>(attn_.get())->decode(
            d_normed, d_attn, session.linearAttnRecurrentStates[type_index_], scratch);
    }
    launch_add(d_hidden, d_attn, d_hidden, hidden, nullptr);

    float *d_post = scratch.token_hidden_b.ensure(hidden, "layer post normed");
    float *d_mlp = scratch.mlp_out_buffer.ensure(hidden, "layer mlp out");
    post_norm_.forward(d_hidden, d_post, 1, hidden);
    mlp_.forward(d_post, d_mlp, 1, hidden, scratch);
    launch_add(d_hidden, d_mlp, d_hidden, hidden, nullptr);

    check_cuda(cudaDeviceSynchronize(), "DecoderLayer decode 同步失败");
}
