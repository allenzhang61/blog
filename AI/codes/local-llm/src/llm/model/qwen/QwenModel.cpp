//
// Created by zhangyoulun on 9/8/2026.
//

#include "QwenModel.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenSession.h"
#include "llm/module/common/RMSNorm.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"

QwenModel::QwenModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling),
      embedding_(weights_.token_embd),
      // tie_word_embeddings=true：LMHead 复用 token_embd 权重。
      lm_head_(weights_.token_embd) {
    const TextConfig &text_config = config_.data.text;
    layers_.reserve(weights_.layers.size());
    for (size_t i = 0; i < weights_.layers.size(); ++i) {
        layers_.emplace_back(weights_.layers[i], text_config);
    }
}

// 在 QwenSession 完整定义可见处生成析构，供 unique_ptr<QwenSession> 正确销毁。
QwenModel::~QwenModel() = default;

int QwenModel::prefill(const Tensor &input) {
    // 为一次新生成重建 session（丢弃上一次请求的 KV cache / recurrent state）。
    session_ = std::make_unique<QwenSession>(config_, input, max_output_tokens_);
    return prefill_session(*session_, input);
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
    session_->output.push_back(token_id);
}

const std::vector<int> &QwenModel::output() const {
    return session_->output;
}

int QwenModel::prefill_session(QwenSession &session, const Tensor &input) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    const int input_size = static_cast<int>(input.numel());
    CudaScratch &scratch = session.scratch;

    // 隐状态 buffer [tokens, hidden]，逐层原位更新。
    Tensor hidden = Tensor::gpu_scratch(
        scratch, scratch_key::kHidden, {static_cast<int64_t>(input_size), static_cast<int64_t>(hidden_size)});
    embedding_.forward(input, hidden, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.prefill(session, hidden);
    }

    // final_norm 仅作用于最后一个 token（下一步预测只需末位隐状态）。
    float *d_hidden = hidden.gpu_f32();
    float *d_last = d_hidden + (input_size - 1) * hidden_size;
    Tensor last = Tensor::gpu_view(d_last, {1, static_cast<int64_t>(hidden_size)});
    Tensor normed = Tensor::gpu_scratch(
        scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)});
    RMSNorm::forward(weights_.output_norm, last, normed,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);

    // lm_head 复用 token_embd（tie），vocab 维度直接取自权重 shape [vocab, hidden]。
    return lm_head_.forward(session, normed, sampler_);
}

int QwenModel::decode_session(QwenSession &session, int prev_token_id, int pos) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    CudaScratch &scratch = session.scratch;

    Tensor hidden = Tensor::gpu_scratch(
        scratch, scratch_key::kHidden, {1, static_cast<int64_t>(hidden_size)});
    const int token_id = prev_token_id;
    Tensor input_view = Tensor::host_view(&token_id, {1}, DType::I32);
    embedding_.forward(input_view, hidden, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.decode(session, hidden, pos);
    }

    Tensor normed = Tensor::gpu_scratch(
        scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)});
    RMSNorm::forward(weights_.output_norm, hidden, normed,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);

    return lm_head_.forward(session, normed, sampler_);
}
