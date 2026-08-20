//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include <cstddef>
#include <utility>

#include "backend/cuda/mem/CudaScratch.h"

DeepseekModel::DeepseekModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling),
      embedding_(*weights_.token_embd),
      lm_head_(*weights_.output) {
    config_.DebugDump();
    mla_layers_.reserve(config_.num_layers);
    mlp_layers_.reserve(config_.num_layers);
    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_.emplace_back(weights_.layers[i], config_);
        mlp_layers_.emplace_back(weights_.layers[i], config_);
    }
}

DeepseekModel::~DeepseekModel() = default;

int DeepseekModel::forward_session(DeepseekSession &session, const Tensor &input, int start_pos) {
    auto &scratch = session.scratch;
    const int input_size = static_cast<int>(input.numel());
    const int hidden_size = config_.hidden_size;

    Tensor hidden = Tensor::gpu_scratch(
        scratch, scratch_key::kHidden, {static_cast<int64_t>(input_size), static_cast<int64_t>(hidden_size)});
    embedding_.forward(input, hidden, scratch);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_[i].forward(session, hidden, start_pos);
        mlp_layers_[i].forward(session, hidden);
    }

    const int last = input_size - 1;
    float *d_hidden = hidden.gpu_f32();
    float *d_last = d_hidden + static_cast<size_t>(last) * hidden_size;
    Tensor last_view = Tensor::gpu_view(d_last, {1, static_cast<int64_t>(hidden_size)});
    Tensor normed = Tensor::gpu_scratch(
        scratch, scratch_key::kNormed, {1, static_cast<int64_t>(hidden_size)});
    RMSNorm::forward(*weights_.output_norm, last_view, normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    return lm_head_.forward(session, normed, sampler_);
}

int DeepseekModel::prefill(const Tensor &input) {
    session_ = std::make_unique<DeepseekSession>(config_, input, max_output_tokens_);
    return forward_session(*session_, input, 0);
}

int DeepseekModel::decode(int prev_token_id, int pos) {
    Tensor input = Tensor::host_view(&prev_token_id, {1}, DType::I32);
    return forward_session(*session_, input, pos);
}

void DeepseekModel::append_output(int token_id) {
    session_->outputs.push_back(token_id);
}

const std::vector<int> &DeepseekModel::outputs() const {
    return session_->outputs;
}

const MemoryUsageProvider &DeepseekModel::memory_usage() const {
    return *session_;
}
