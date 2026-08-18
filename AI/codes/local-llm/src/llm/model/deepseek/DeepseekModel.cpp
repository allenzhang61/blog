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
      embedding_(*weights_.token_embd, &global_cuda_weight_pool()),
      lm_head_(*weights_.output, &global_cuda_weight_pool()) {
    config_.DebugDump();
    mla_layers_.reserve(config_.num_layers);
    mlp_layers_.reserve(config_.num_layers);
    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_.emplace_back(weights_.layers[i], config_, &global_cuda_weight_pool());
        mlp_layers_.emplace_back(weights_.layers[i], config_, &global_cuda_weight_pool());
    }
}

DeepseekModel::~DeepseekModel() = default;

int DeepseekModel::forward_session(DeepseekSession &session, const std::vector<int> &input, int start_pos) {
    auto &scratch = session.scratch;
    const int input_size = static_cast<int>(input.size());
    const int hidden_size = config_.hidden_size;
    const int vocab_size = config_.vocab_size;

    float *d_hidden = scratch.ensure<float>(scratch_key::kHidden, input_size * hidden_size);
    embedding_.forward(input, d_hidden, scratch);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_[i].forward(session, d_hidden, input_size, start_pos);
        mlp_layers_[i].forward(session, d_hidden, input_size);
    }

    const int last = input_size - 1;
    float *d_last = d_hidden + static_cast<size_t>(last) * hidden_size;
    float *d_normed = scratch.ensure<float>(scratch_key::kNormed, static_cast<size_t>(hidden_size));
    RMSNorm::forward(&global_cuda_weight_pool(), *weights_.output_norm, d_last, d_normed, 1,
                     hidden_size, config_.rms_norm_eps, /*one_plus=*/false);

    return lm_head_.forward(session, d_normed, hidden_size, vocab_size, sampler_);
}

int DeepseekModel::prefill(const std::vector<int> &input) {
    session_ = std::make_unique<DeepseekSession>(config_, input, max_output_tokens_);
    return forward_session(*session_, input, 0);
}

int DeepseekModel::decode(int prev_token_id, int pos) {
    return forward_session(*session_, {prev_token_id}, pos);
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
