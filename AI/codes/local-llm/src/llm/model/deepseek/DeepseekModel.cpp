//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include <cstddef>
#include <utility>

#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"

DeepseekModel::DeepseekModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling) {
    config_.DebugDump();
    mla_layers_.reserve(config_.num_layers);
    mlp_layers_.reserve(config_.num_layers);
    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_.emplace_back(weights_.layers[i], config_);
        mlp_layers_.emplace_back(weights_.layers[i], config_);
    }
}

DeepseekModel::~DeepseekModel() = default;

int DeepseekModel::forward_session(DeepseekSession &session, const CPUTensor &c_input_i32, int start_pos) {
    auto &scratch = session.scratch;
    const int input_size = static_cast<int>(c_input_i32.numel());
    const int hidden_size = config_.hidden_size;

    GPUTensor g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {static_cast<int64_t>(input_size), static_cast<int64_t>(hidden_size)},
        DType::F32);
    embedding_.forward(*weights_.s_token_embd, c_input_i32, g_hidden_f32, scratch);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_[i].forward(session, g_hidden_f32, start_pos);
        mlp_layers_[i].forward(session, g_hidden_f32);
    }

    const int last = input_size - 1;
    const size_t last_offset = static_cast<size_t>(last) * hidden_size * sizeof(float);
    GPUTensor g_last_view = GPUTensor(g_hidden_f32, last_offset, {1, static_cast<int64_t>(hidden_size)});
    GPUTensor g_normed = GPUTensor(
        scratch, scratch_key::kNormed, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_last_view, g_normed,
                     config_.rms_norm_eps, /*one_plus=*/false);

    return lm_head_.forward(*weights_.s_output, session, g_normed, sampler_);
}

int DeepseekModel::prefill(const CPUTensor &c_input) {
    session_ = std::make_unique<DeepseekSession>(config_, c_input, max_output_tokens_);
    return forward_session(*session_, c_input, 0);
}

int DeepseekModel::decode(int prev_token_id, int pos) {
    CPUTensor c_input = CPUTensor(&prev_token_id, {1}, DType::I32);
    return forward_session(*session_, c_input, pos);
}

void DeepseekModel::append_output(int token_id) {
    session_->output.push_back(token_id);
}

const std::vector<int> &DeepseekModel::output() const {
    return session_->output;
}

const MemoryUsageProvider &DeepseekModel::memory_usage() const {
    return *session_;
}
