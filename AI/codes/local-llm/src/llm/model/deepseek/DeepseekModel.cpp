//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"
#include "utils/stats/ScopedTimer.h"

namespace {
    bool deepseek_trace_enabled() {
        const char *env = std::getenv("LOCAL_LLM_DEEPSEEK_TRACE");
        return env != nullptr && std::atoi(env) > 0;
    }

    bool deepseek_device_argmax_disabled() {
        const char *env = std::getenv("LOCAL_LLM_DISABLE_DEEPSEEK_DEVICE_ARGMAX");
        return env != nullptr && std::atoi(env) > 0;
    }

    bool deepseek_trace_pos_match(int pos) {
        const char *env = std::getenv("LOCAL_LLM_DEEPSEEK_TRACE_POS");
        if (env == nullptr || env[0] == '\0') return true;
        return pos == std::atoi(env);
    }

    void trace_hidden(DeepseekSession &session, const GPUTensor &g_hidden_f32,
                      const char *stage, int pos, int layer) {
        if (!deepseek_trace_enabled() || !deepseek_trace_pos_match(pos)) return;
        CPUTensor c_hidden = g_hidden_f32.to_host(session.cpu_scratch, "ds.trace.hidden", "ds.trace.hidden");
        const float *h = c_hidden.data<float>();
        const int64_t n = c_hidden.numel();
        double sum = 0.0;
        double sum_abs = 0.0;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float v = h[i];
            sum += v;
            sum_abs += std::fabs(static_cast<double>(v));
            max_abs = std::max(max_abs, std::fabs(v));
        }
        std::cerr << "[ds.trace] pos=" << pos
                  << " layer=" << layer
                  << " stage=" << stage
                  << " n=" << n
                  << " sum=" << sum
                  << " mean_abs=" << (n > 0 ? sum_abs / static_cast<double>(n) : 0.0)
                  << " max_abs=" << max_abs
                  << " first=";
        const int64_t limit = std::min<int64_t>(8, n);
        for (int64_t i = 0; i < limit; ++i) {
            if (i) std::cerr << ",";
            std::cerr << h[i];
        }
        std::cerr << "\n";
    }
} // namespace

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

    if (!sampler_.is_greedy() || deepseek_device_argmax_disabled()) {
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
    const auto g_input_i32 = GPUTensor(deepseek_session.d_token(), {1}, DType::I32, "deepseek.decode.token");
    TensorTool::embedding_lookup(*weights_.s_token_embd, g_input_i32, g_hidden_f32, stream);
    trace_hidden(deepseek_session, g_hidden_f32, "embedding", pos, -1);

    for (int i = 0; i < config_.num_layers; ++i) {
        {
            ScopedCpuTimer t("ds.decode.mla_forward");
            mla_layers_[i].forward(deepseek_session, g_hidden_f32, pos);
        }
        trace_hidden(deepseek_session, g_hidden_f32, "mla", pos, i);
        {
            ScopedCpuTimer t("ds.decode.mlp_forward");
            ScopedCpuTimer split_t(weights_.layers[i].is_moe
                                       ? "ds.decode.mlp_moe_forward"
                                       : "ds.decode.mlp_dense_forward");
            mlp_layers_[i].forward(deepseek_session, g_hidden_f32);
        }
        trace_hidden(deepseek_session, g_hidden_f32, "mlp", pos, i);
    }

    const auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {1, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    trace_hidden(deepseek_session, g_normed_f32, "final_norm", pos, config_.num_layers);
    {
        ScopedCpuTimer t("ds.decode.lm_head_argmax_device");
        lm_head_.forward_argmax_device(*weights_.s_output, deepseek_session, g_normed_f32,
                                       deepseek_session.d_token(), stream);
    }

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
