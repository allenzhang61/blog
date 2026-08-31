//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <utility>

#include "backend/cuda/common.h"
#include "backend/cuda/graph/CudaGraph.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekRuntimeOptions.h"
#include "llm/model/deepseek/DeepseekTrace.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"
#include "utils/stats/Profiler.h"
#include "utils/stats/ScopedTimer.h"

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
    const int trace_pos = start_pos + static_cast<int>(input_size) - 1;
    deepseek_trace::tensor(session, g_hidden_f32, "embedding", trace_pos, -1);

    for (int i = 0; i < config_.num_layers; ++i) {
        session.trace_pos = trace_pos;
        session.trace_layer = i;
        mla_layers_[i].forward(session, g_hidden_f32, start_pos);
        deepseek_trace::tensor(session, g_hidden_f32, "mla", trace_pos, i);
        mlp_layers_[i].forward(session, g_hidden_f32);
        deepseek_trace::tensor(session, g_hidden_f32, "mlp", trace_pos, i);
    }
    session.trace_pos = -1;
    session.trace_layer = -1;

    const int64_t last = input_size - 1;
    const size_t last_offset = static_cast<size_t>(last) * hidden_size * sizeof(float);
    const auto g_last_view_f32 = GPUTensor(g_hidden_f32, last_offset, {1, hidden_size});
    const auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {1, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_last_view_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    deepseek_trace::tensor(session, g_normed_f32, "final_norm", trace_pos, config_.num_layers);

    session.trace_pos = trace_pos;
    session.trace_layer = config_.num_layers;
    const int next = lm_head_.forward(*weights_.s_output, session, g_normed_f32, sampler_);
    session.trace_pos = -1;
    session.trace_layer = -1;
    return next;
}

SessionBase *DeepseekModel::create_session(const std::string &text) {
    return new DeepseekSession(config_, encode_text(text), max_output_tokens_);
}

int DeepseekModel::prefill(SessionBase &session) {
    return forward_session(dynamic_cast<DeepseekSession &>(session), session.h_input_i32_, 0);
}

int DeepseekModel::decode(SessionBase &session_base) {
    auto &session = dynamic_cast<DeepseekSession &>(session_base);
    const int prev_token_id = session.prev_token_id();
    const int pos = session.decode_pos();

    if (!sampler_.is_greedy()) {
        //temperature>0会进入这个分支
        const auto c_input_i32 = CPUTensor(&prev_token_id, {1}, DType::I32);
        return forward_session(session, c_input_i32, pos);
    }

    session.d_token().set_data(prev_token_id, "deepseek decode token H2D 失败");
    session.d_pos().set_data(pos, "deepseek decode pos H2D 失败");

    const bool graph_enabled = deepseek_runtime_options().cuda_graph;
    const bool profile_this_step = Profiler::instance().capturing();
    if (!graph_enabled) {
        eager_decode_greedy_device(session, pos);
        const CPUTensor h_next_token_i32 = session.d_token().to_host(
            session.cpu_scratch, "deepseek.decode.token",
            "deepseek decode token D2H 失败");
        return h_next_token_i32.data<int>()[0];
    }

    // 如果是当前 session 的第一个 greedy decode step，或者 profiler 正在采样这一 step，就走 eager：
    if (session.decode_greedy_steps == 0 || profile_this_step) {
        eager_decode_greedy_device(session, pos);
        // 如果这一步不是 profiler step，就把已有 graph 标记为失效。原因是刚才 eager 可能让 scratch buffer 扩容，
        //      旧 graph 里捕获的指针可能不再可靠，所以下一步要重新录。
        // 如果是 profiler step，则不标记 stale。因为 profiler 只是临时绕开 graph，为了让逐 kernel 计时能被 ScopedTimer 抓到；
        //      它不应该破坏已经录好的 graph。
        if (!profile_this_step) {
            mark_cuda_graph_stale(session.decode_graph);
        }
    } else {
        // 如果 CUDA Graph 还没准备好，就录一次 graph。record_decode_graph(session) 会把单步 decode 的 GPU 计算录下来
        if (!session.decode_graph.ready()) {
            record_decode_graph(session);
        }
        // 后续 decode step 就 replay 这个 graph，减少大量 kernel launch 开销。
        launch_cuda_graph(session.decode_graph, "deepseek decode graph launch 失败");
    }
    // 记录这个 session 已经跑过一个 greedy decode step。这样下次不会再按“第一步”处理。
    ++session.decode_greedy_steps;
    // 更新每一层 KV cache 的有效长度。当前 decode 处理的位置是 pos，所以执行后有效序列长度变成：
    // pos + 1
    // 这个很重要，后面的 attention 要知道历史 cache 到哪里是有效的。
    for (auto &kv: session.kv_caches) {
        kv.seq_len = pos + 1;
    }

    const CPUTensor h_next_token_i32 = session.d_token().to_host(
        session.cpu_scratch, "deepseek.decode.token",
        "deepseek decode token D2H 失败");
    return h_next_token_i32.data<int>()[0];
}

/*
*   eager     = 直接逐个 kernel 执行，不走 CUDA Graph replay
    decode    = 生成阶段每次只处理 1 个 token
    greedy    = 只取概率最大的 token，也就是 argmax
    device    = token 输入、argmax 输出尽量都留在 GPU 上
 *
 */
void DeepseekModel::eager_decode_greedy_device(DeepseekSession &session, int pos) {
    auto &scratch = session.cuda_scratch;
    const int64_t hidden_size = config_.hidden_size;
    const auto g_hidden_f32 = GPUTensor(scratch, scratch_key::kHidden, {1, hidden_size}, DType::F32);
    const GPUTensor &g_input_i32 = session.d_token();
    TensorTool::embedding_lookup(*weights_.s_token_embd, g_input_i32, g_hidden_f32);
    deepseek_trace::tensor(session, g_hidden_f32, "embedding", pos, -1);

    for (int i = 0; i < config_.num_layers; ++i) {
        session.trace_pos = pos;
        session.trace_layer = i;
        {
            ScopedCpuTimer t("ds.decode.mla_forward");
            mla_layers_[i].forward(session, g_hidden_f32, pos, /*use_device_pos=*/true);
        }
        deepseek_trace::tensor(session, g_hidden_f32, "mla", pos, i);
        {
            ScopedCpuTimer t("ds.decode.mlp_forward");
            ScopedCpuTimer split_t(weights_.layers[i].is_moe
                                       ? "ds.decode.mlp_moe_forward"
                                       : "ds.decode.mlp_dense_forward");
            mlp_layers_[i].forward(session, g_hidden_f32);
        }
        deepseek_trace::tensor(session, g_hidden_f32, "mlp", pos, i);
    }
    session.trace_pos = -1;
    session.trace_layer = -1;

    const auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {1, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    deepseek_trace::tensor(session, g_normed_f32, "final_norm", pos, config_.num_layers);
    {
        ScopedCpuTimer t("ds.decode.lm_head_argmax_device");
        session.trace_pos = pos;
        session.trace_layer = config_.num_layers;
        lm_head_.forward_argmax_device(*weights_.s_output, session, g_normed_f32,
                                       session.d_token());
        session.trace_pos = -1;
        session.trace_layer = -1;
    }
}

void DeepseekModel::record_decode_graph(DeepseekSession &deepseek_session) {
    auto &scratch = deepseek_session.cuda_scratch;
    const int64_t hidden_size = config_.hidden_size;

    destroy_cuda_graph(deepseek_session.decode_graph);
    begin_thread_local_cuda_graph_capture("deepseek decode graph begin capture 失败");

    const auto g_hidden_f32 = GPUTensor(scratch, scratch_key::kHidden, {1, hidden_size}, DType::F32);
    const GPUTensor &g_input_i32 = deepseek_session.d_token();
    TensorTool::embedding_lookup(*weights_.s_token_embd, g_input_i32, g_hidden_f32);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_layers_[i].forward(deepseek_session, g_hidden_f32, /*start_pos=*/0, /*use_device_pos=*/true);
        mlp_layers_[i].forward(deepseek_session, g_hidden_f32);
    }

    const auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {1, hidden_size}, DType::F32);
    RMSNorm::forward(*weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    lm_head_.forward_argmax_device(*weights_.s_output, deepseek_session, g_normed_f32,
                                   deepseek_session.d_token());

    end_cuda_graph_capture_and_instantiate(deepseek_session.decode_graph,
                                           "deepseek decode graph capture/instantiate 失败");
}

std::string DeepseekModel::output(const SessionBase &session) const {
    return decode_text(session.h_output_i32_);
}

const MemoryUsageProvider &DeepseekModel::memory_usage(const SessionBase &session) const {
    return session;
}
