//
// Created by zhangyoulun on 9/8/2026.
//

#include "QwenModel.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenSession.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"

namespace {
const TextConfig &text_of(const QwenConfig &config) { return config.data.text; }
} // namespace

QwenModel::QwenModel(const QwenConfig &config, const QwenWeights &weights, CudaWeightPool *pool)
    : config_(config),
      pool_(pool),
      embed_tokens_(weights.embed_tokens, pool),
      final_norm_(weights.final_norm, pool, config.data.text.rms_norm_eps),
      // tie_word_embeddings=true：LMHead 复用 embed_tokens 权重。
      lm_head_(weights.embed_tokens, pool) {
    const TextConfig &t = text_of(config);
    layers_.reserve(weights.layers.size());
    for (int i = 0; i < static_cast<int>(weights.layers.size()); ++i) {
        layers_.emplace_back(weights.layers[i], t, pool, i);
    }
}

int QwenModel::prefill(QwenSession &session, const std::vector<int> &h_input_ids) {
    const TextConfig &t = text_of(config_);
    const int hidden = t.hidden_size;
    const int tokens = static_cast<int>(h_input_ids.size());
    QwenForwardScratch &scratch = session.forwardScratch;

    // 隐状态 buffer [tokens, hidden]，逐层原位更新。
    float *d_hidden = scratch.layer_out_buffer.ensure(static_cast<size_t>(tokens) * hidden, "prefill hidden");
    embed_tokens_.forward(h_input_ids, d_hidden, hidden);

    for (DecoderLayer &layer : layers_) {
        layer.prefill(d_hidden, tokens, session, scratch);
    }

    // final_norm 仅作用于最后一个 token（下一步预测只需末位隐状态）。
    float *d_last = d_hidden + static_cast<size_t>(tokens - 1) * hidden;
    float *d_normed = scratch.token_hidden_a.ensure(hidden, "prefill final normed");
    final_norm_.forward(d_last, d_normed, 1, hidden);

    return lm_head_.forward(d_normed, hidden, scratch);
}

int QwenModel::decode(QwenSession &session, int prev_token_id, int pos) {
    const TextConfig &t = text_of(config_);
    const int hidden = t.hidden_size;
    QwenForwardScratch &scratch = session.forwardScratch;

    float *d_hidden = scratch.layer_out_buffer.ensure(hidden, "decode hidden");
    embed_tokens_.forward(std::vector<int>{prev_token_id}, d_hidden, hidden);

    for (DecoderLayer &layer : layers_) {
        layer.decode(d_hidden, pos, session, scratch);
    }

    float *d_normed = scratch.token_hidden_a.ensure(hidden, "decode final normed");
    final_norm_.forward(d_hidden, d_normed, 1, hidden);

    return lm_head_.forward(d_normed, hidden, scratch);
}
