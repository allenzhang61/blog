//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

DeepseekModel::DeepseekModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling) {
    config_.debug_dump();
    mla_layers_.reserve(config_.num_layers);
    mlp_layers_.reserve(config_.num_layers);
    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_.emplace_back(weights_.layers[i], config_);
        mlp_layers_.emplace_back(weights_.layers[i], config_);
    }
}

DeepseekModel::~DeepseekModel() = default;

int DeepseekModel::forward_session(DeepseekSession &session, const CPUTensor &c_input_i32, const int start_pos) {
    auto &scratch = session.cuda_scratch;
    const int64_t input_size = c_input_i32.numel();
    const int64_t hidden_size = config_.hidden_size;

    const auto g_hidden_f32 = GPUTensor(scratch, scratch_key::kHidden, {input_size, hidden_size}, DType::F32);
    embedding_.forward(*weights_.s_token_embd, c_input_i32, g_hidden_f32, scratch);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_[i].forward(session, g_hidden_f32, start_pos);
        mlp_layers_[i].forward(session, g_hidden_f32);
    }

    const int64_t last = input_size - 1;
    const size_t last_offset = static_cast<size_t>(last) * hidden_size * sizeof(float);
    const auto g_last_view_f32 = GPUTensor(g_hidden_f32, last_offset, {1, hidden_size});
    const auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {1, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_last_view_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);

    return lm_head_.forward(*weights_.s_output, session, g_normed_f32, sampler_);
}

SessionBase *DeepseekModel::create_session(const std::string &text) {
    return new DeepseekSession(config_, encode_text(text), max_output_tokens_);
}

int DeepseekModel::prefill(SessionBase &session) {
    return forward_session(dynamic_cast<DeepseekSession &>(session), session.h_input_i32_, 0);
}

int DeepseekModel::decode(SessionBase &session) {
    auto &deepseek_session = dynamic_cast<DeepseekSession &>(session);
    const int prev_token_id = deepseek_session.prev_token_id();
    const int pos = deepseek_session.decode_pos();

    if (!sampler_.is_greedy()) {
        const auto c_input_i32 = CPUTensor(&prev_token_id, {1}, DType::I32);
        return forward_session(deepseek_session, c_input_i32, pos);
    }

    cudaStream_t stream = get_current_cuda_stream();
    check_cuda(cudaMemcpyAsync(deepseek_session.d_token(), &prev_token_id, sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "deepseek decode token H2D 失败");

    auto &scratch = deepseek_session.cuda_scratch;
    const int64_t hidden_size = config_.hidden_size;
    const auto g_hidden_f32 = GPUTensor(scratch, scratch_key::kHidden, {1, hidden_size}, DType::F32);
    TensorTool::embedding_lookup_device(*weights_.s_token_embd, deepseek_session.d_token(), g_hidden_f32, stream);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_[i].forward(deepseek_session, g_hidden_f32, pos);
        mlp_layers_[i].forward(deepseek_session, g_hidden_f32);
    }

    const auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {1, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    lm_head_.forward_argmax_device(*weights_.s_output, deepseek_session, g_normed_f32,
                                   deepseek_session.d_token(), stream);

    int next_token = 0;
    cuda_memcpy_d2h(&next_token, deepseek_session.d_token(), sizeof(int), "deepseek decode token D2H 失败");
    return next_token;
}

std::string DeepseekModel::output(const SessionBase &session) const {
    return decode_text(session.h_output_i32_);
}

const MemoryUsageProvider &DeepseekModel::memory_usage(const SessionBase &session) const {
    return session;
}
