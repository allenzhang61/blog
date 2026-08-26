//
// Created by zhangyoulun on 9/8/2026.
//

#include "QwenModel.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenSession.h"
#include "llm/module/common/RMSNorm.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/common.h"
#include "backend/cuda/graph/CudaGraph.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"
#include "utils/stats/Profiler.h"

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

QwenModel::~QwenModel() = default;

SessionBase *QwenModel::create_session(const std::string &text) {
    return new QwenSession(config_, encode_text(text), max_output_tokens_);
}

int QwenModel::prefill(SessionBase &session) {
    auto &qwen_session = dynamic_cast<QwenSession &>(session);
    const CPUTensor &c_input_i32 = qwen_session.h_input_i32_;
    const TextConfig &text_config = config_.data.text;
    const int64_t hidden_size = text_config.hidden_size;
    const int64_t input_size = c_input_i32.numel();
    CudaScratch &scratch = qwen_session.cuda_scratch;

    // 隐状态 buffer [tokens, g_hidden]，逐层原位更新。
    GPUTensor g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {input_size, hidden_size},
        DType::F32);
    embedding_.forward(weights_.s_token_embd, c_input_i32, g_hidden_f32, scratch);

    for (DecoderLayer &layer : layers_) {
        layer.prefill(qwen_session, g_hidden_f32);
    }

    // final_norm 仅作用于最后一个 token（下一步预测只需末位隐状态）。
    const size_t last_offset = static_cast<size_t>(input_size - 1) * hidden_size * sizeof(float);
    GPUTensor g_last_f32 = GPUTensor(g_hidden_f32, last_offset, {1, hidden_size});
    GPUTensor g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, hidden_size}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_last_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);

    // lm_head 复用 token_embd（tie），vocab 维度直接取自权重 shape [vocab, g_hidden]。
    return lm_head_.forward(weights_.s_token_embd, qwen_session, g_normed_f32, sampler_);
}

int QwenModel::decode(SessionBase &session) {
    auto &qwen_session = dynamic_cast<QwenSession &>(session);
    const int prev_token_id = qwen_session.prev_token_id();
    const int pos = qwen_session.decode_pos();

    // 非贪心（温度/top-k/top-p/重复惩罚）需 host 端 Sampler，无法走 GPU argmax 闭环，
    // 退回原逐 kernel launch 路径。
    if (!sampler_.is_greedy()) {
        const TextConfig &text_config = config_.data.text;
        const int64_t hidden_size = text_config.hidden_size;
        CudaScratch &scratch = qwen_session.cuda_scratch;

        const auto g_hidden_f32 = GPUTensor(
            scratch, scratch_key::kHidden, {1, hidden_size}, DType::F32);
        const int token_id = prev_token_id;
        const auto c_input_view_i32 = CPUTensor(&token_id, {1}, DType::I32);
        embedding_.forward(weights_.s_token_embd, c_input_view_i32, g_hidden_f32, scratch);
        for (DecoderLayer &layer : layers_) layer.decode(qwen_session, g_hidden_f32, pos);
        const auto g_normed_f32 = GPUTensor(
            scratch, scratch_key::kTokenHiddenA, {1, hidden_size}, DType::F32);
        RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                         config_.data.text.rms_norm_eps, /*one_plus=*/true);
        return lm_head_.forward(weights_.s_token_embd, qwen_session, g_normed_f32, sampler_);
    }

    // === 贪心：CUDA Graph 路径 ===
    cudaStream_t stream = get_current_cuda_stream();
    // 每步 graph 外把 host 侧的 pos / 输入 token 异步写入 device buffer（graph 内 kernel 只读它们）。
    check_cuda(cudaMemcpyAsync(qwen_session.d_pos(), &pos, sizeof(int), cudaMemcpyHostToDevice, stream),
               "decode pos H2D 失败");
    check_cuda(cudaMemcpyAsync(qwen_session.d_token(), &prev_token_id, sizeof(int), cudaMemcpyHostToDevice, stream),
               "decode token H2D 失败");

    // profile 采样步走 eager：graph replay 内的 kernel 不经 ScopedGpuTimer 埋点，
    // 无法拿到逐 kernel 细分；此步改走 eager 让 profiler 抓到每个 kernel 的耗时。
    // 非采样步仍走 graph，保持稳态吞吐口径不变。
    const bool profile_this_step = Profiler::instance().capturing();

    if (qwen_session.decode_greedy_steps == 0 || profile_this_step) {
        // 首步（或换 session / profile 采样步）走 eager：把所有 grow-only scratch 撑到
        // decode 稳态尺寸，避免随后 capture 期间触发非法的 cudaMalloc；同时 GPU argmax 把结果写回 d_token。
        eager_decode_greedy_device(qwen_session);
        // profile 采样步不使已建好的 graph 失效（它没改 scratch 尺寸，下步可继续 replay）。
        if (!profile_this_step) {
            mark_cuda_graph_stale(qwen_session.decode_graph); // scratch 刚可能增长，之前若有 graph 也失效
        }
    } else {
        if (!qwen_session.decode_graph.ready()) {
            record_decode_graph(qwen_session);
        }
        launch_cuda_graph(qwen_session.decode_graph, stream, "decode graph launch 失败");
    }
    ++qwen_session.decode_greedy_steps;

    // 取回 argmax 得到的下一个 token id（graph 外的一次 int D2H + 同流同步）。
    int next_token = 0;
    cuda_memcpy_d2h(&next_token, qwen_session.d_token(), sizeof(int), "decode token D2H 失败");
    return next_token;
}

std::string QwenModel::output(const SessionBase &session) const {
    return decode_text(session.h_output_i32_);
}

const MemoryUsageProvider &QwenModel::memory_usage(const SessionBase &session) const {
    return session;
}

void QwenModel::eager_decode_greedy_device(QwenSession &session) {
    const TextConfig &text_config = config_.data.text;
    const int64_t hidden_size = text_config.hidden_size;
    CudaScratch &scratch = session.cuda_scratch;
    cudaStream_t stream = get_current_cuda_stream();

    const auto g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {1, hidden_size}, DType::F32);
    TensorTool::embedding_lookup_device(weights_.s_token_embd, session.d_token(), g_hidden_f32, stream);
    for (DecoderLayer &layer : layers_) layer.decode(session, g_hidden_f32, /*pos=*/0);
    const auto g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, hidden_size}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);
    lm_head_.forward_argmax_device(weights_.s_token_embd, session, g_normed_f32, session.d_token(), stream);
}

void QwenModel::record_decode_graph(QwenSession &session) {
    const TextConfig &text_config = config_.data.text;
    const int64_t hidden_size = text_config.hidden_size;
    CudaScratch &scratch = session.cuda_scratch;
    cudaStream_t stream = get_current_cuda_stream();

    // 若之前有旧 graph（换 session 重建），先销毁，再开始 capture。
    destroy_cuda_graph(session.decode_graph);

    // ThreadLocal 模式：仅捕获本线程当前流上的操作，避免误捕获库内部的其它流活动。
    begin_thread_local_cuda_graph_capture(stream, "decode graph begin capture 失败");

    const auto g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {1, hidden_size}, DType::F32);
    // embedding 从 device token buffer 读输入 token（不做 H2D）。
    TensorTool::embedding_lookup_device(weights_.s_token_embd, session.d_token(), g_hidden_f32, stream);

    for (DecoderLayer &layer : layers_) {
        // pos 走 session.d_pos()（device），此处 host pos 仅用于层内已废弃的 seq_len 记账，传 0 即可。
        layer.decode(session, g_hidden_f32, /*pos=*/0);
    }

    const auto g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, hidden_size}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);
    // lm_head GEMM + GPU argmax，把下一 token 写回 device token buffer（闭环，无 D2H）。
    lm_head_.forward_argmax_device(weights_.s_token_embd, session, g_normed_f32, session.d_token(), stream);

    end_cuda_graph_capture_and_instantiate(stream, session.decode_graph, "decode graph capture/instantiate 失败");
}
