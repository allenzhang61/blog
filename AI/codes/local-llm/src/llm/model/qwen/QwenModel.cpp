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
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"

QwenModel::QwenModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling) {
    const TextConfig &text_config = config_.data.text;
    layers_.reserve(weights_.layers.size());
    for (size_t i = 0; i < weights_.layers.size(); ++i) {
        layers_.emplace_back(weights_.layers[i], text_config);
    }
}

// 在 QwenSession 完整定义可见处生成析构，供 unique_ptr<QwenSession> 正确销毁。
QwenModel::~QwenModel() = default;

int QwenModel::prefill(const CPUTensor &c_input_i32) {
    // 为一次新生成重建 session（丢弃上一次请求的 KV cache / recurrent state）。
    session_ = std::make_unique<QwenSession>(config_, c_input_i32, max_output_tokens_);
    return prefill_session(*session_, c_input_i32);
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

int QwenModel::prefill_session(QwenSession &session, const CPUTensor &c_input_i32) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    const int input_size = static_cast<int>(c_input_i32.numel());
    CudaScratch &scratch = session.scratch;

    // 隐状态 buffer [tokens, g_hidden]，逐层原位更新。
    GPUTensor g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {static_cast<int64_t>(input_size), static_cast<int64_t>(hidden_size)},
        DType::F32);
    embedding_.forward(weights_.s_token_embd, c_input_i32, g_hidden_f32, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.prefill(session, g_hidden_f32);
    }

    // final_norm 仅作用于最后一个 token（下一步预测只需末位隐状态）。
    const size_t last_offset = static_cast<size_t>(input_size - 1) * hidden_size * sizeof(float);
    GPUTensor g_last_f32 = GPUTensor(g_hidden_f32, last_offset, {1, static_cast<int64_t>(hidden_size)});
    GPUTensor g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_last_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);

    // lm_head 复用 token_embd（tie），vocab 维度直接取自权重 shape [vocab, g_hidden]。
    return lm_head_.forward(weights_.s_token_embd, session, g_normed_f32, sampler_);
}

int QwenModel::decode_session(QwenSession &session, int prev_token_id, int pos) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    CudaScratch &scratch = session.scratch;

    GPUTensor g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    const int token_id = prev_token_id;
    CPUTensor c_input_view_i32 = CPUTensor(&token_id, {1}, DType::I32);
    embedding_.forward(weights_.s_token_embd, c_input_view_i32, g_hidden_f32, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.decode(session, g_hidden_f32, pos);
    }

    GPUTensor g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);

    return lm_head_.forward(weights_.s_token_embd, session, g_normed_f32, sampler_);
}
