//
// Created by zhangyoulun on 9/8/2026.
//

#include "QwenModel.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenSession.h"
#include "llm/model/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"

QwenModel::QwenModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling),
      embed_tokens_(weights_.embed_tokens, &global_cuda_weight_pool()),
      final_norm_(weights_.final_norm, &global_cuda_weight_pool(), config_.data.text.rms_norm_eps),
      // tie_word_embeddings=true：LMHead 复用 embed_tokens 权重。
      lm_head_(weights_.embed_tokens, &global_cuda_weight_pool()) {
    const TextConfig &text_config = config_.data.text;
    layers_.reserve(weights_.layers.size());
    for (size_t i = 0; i < weights_.layers.size(); ++i) {
        layers_.emplace_back(weights_.layers[i], text_config, &global_cuda_weight_pool(), i);
    }
}

// 在 QwenSession 完整定义可见处生成析构，供 unique_ptr<QwenSession> 正确销毁。
QwenModel::~QwenModel() = default;

int QwenModel::prefill(const std::vector<int> &inputs) {
    // 为一次新生成重建 session（丢弃上一次请求的 KV cache / recurrent state）。
    session_ = std::make_unique<QwenSession>(config_, inputs, max_output_tokens_);
    return prefill_session(*session_, inputs);
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
    session_->outputs.push_back(token_id);
}

const std::vector<int> &QwenModel::outputs() const {
    return session_->outputs;
}

int QwenModel::prefill_session(QwenSession &session, const std::vector<int> &inputs) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    const size_t input_size = inputs.size();
    QwenForwardScratch &scratch = session.forwardScratch;

    // 隐状态 buffer [tokens, hidden]，逐层原位更新。
    float *d_hidden = scratch.layer_out_buffer.ensure(input_size * static_cast<size_t>(hidden_size), "prefill hidden");
    embed_tokens_.forward(inputs, d_hidden, hidden_size, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.prefill(d_hidden, input_size, session, scratch);
    }

    // final_norm 仅作用于最后一个 token（下一步预测只需末位隐状态）。
    float *d_last = d_hidden + (input_size - 1) * static_cast<size_t>(hidden_size);
    float *d_normed = scratch.token_hidden_a.ensure(hidden_size, "prefill final normed");
    final_norm_.forward(d_last, d_normed, 1, hidden_size, scratch);

    return lm_head_.forward(d_normed, hidden_size, scratch, sampler_, session.outputs);
}

int QwenModel::decode_session(QwenSession &session, int prev_token_id, int pos) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    QwenForwardScratch &scratch = session.forwardScratch;

    float *d_hidden = scratch.layer_out_buffer.ensure(hidden_size, "decode hidden");
    embed_tokens_.forward(std::vector<int>{prev_token_id}, d_hidden, hidden_size, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.decode(d_hidden, pos, session, scratch);
    }

    float *d_normed = scratch.token_hidden_a.ensure(hidden_size, "decode final normed");
    final_norm_.forward(d_hidden, d_normed, 1, hidden_size, scratch);

    return lm_head_.forward(d_normed, hidden_size, scratch, sampler_, session.outputs);
}
