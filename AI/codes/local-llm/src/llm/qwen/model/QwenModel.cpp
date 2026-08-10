//
// Created by zhangyoulun on 9/8/2026.
//

#include "QwenModel.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenSession.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"

namespace {
const TextConfig &text_of(const QwenConfig &config) { return config.data.text; }
} // namespace

QwenModel::QwenModel(const std::string &model_dir, int max_output_tokens)
    : config_(model_dir + "/config.json"),
      weights_(model_dir, config_),
      tokenizer_(model_dir + "/tokenizer.json"),
      max_output_tokens_(max_output_tokens),
      embed_tokens_(weights_.embed_tokens, &pool_),
      final_norm_(weights_.final_norm, &pool_, config_.data.text.rms_norm_eps),
      // tie_word_embeddings=true：LMHead 复用 embed_tokens 权重。
      lm_head_(weights_.embed_tokens, &pool_) {
    const TextConfig &t = text_of(config_);
    layers_.reserve(weights_.layers.size());
    for (int i = 0; i < static_cast<int>(weights_.layers.size()); ++i) {
        layers_.emplace_back(weights_.layers[i], t, &pool_, i);
    }
}

// 在 QwenSession 完整定义可见处生成析构，供 unique_ptr<QwenSession> 正确销毁。
QwenModel::~QwenModel() = default;

int QwenModel::prefill(const std::vector<int> &input_ids) {
    // 为一次新生成重建 session（丢弃上一次请求的 KV cache / recurrent state）。
    session_ = std::make_unique<QwenSession>(config_, input_ids, max_output_tokens_);
    return prefill_session(*session_, input_ids);
}

int QwenModel::decode(int prev_token_id, int pos) {
    if (!session_) {
        throw std::runtime_error("QwenModel::decode 在 prefill 之前被调用");
    }
    return decode_session(*session_, prev_token_id, pos);
}

const MemoryUsageProvider &QwenModel::memory_usage() const {
    if (!session_) {
        throw std::runtime_error("QwenModel::memory_usage 在 prefill 之前被调用");
    }
    return *session_;
}

void QwenModel::append_output(int token_id) {
    session_->h_outputs.push_back(token_id);
}

const std::vector<int> &QwenModel::outputs() const {
    return session_->h_outputs;
}

int QwenModel::prefill_session(QwenSession &session, const std::vector<int> &h_input_ids) {
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

int QwenModel::decode_session(QwenSession &session, int prev_token_id, int pos) {
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
