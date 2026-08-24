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
#include "backend/cuda/common.h"
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

// 在 QwenSession 完整定义可见处生成析构，供 unique_ptr<QwenSession> 正确销毁。
QwenModel::~QwenModel() {
    // 释放 decode CUDA Graph 资源（capture 出来的 graph 与实例化的可执行图）。
    if (decode_graph_exec_) cudaGraphExecDestroy(decode_graph_exec_);
    if (decode_graph_) cudaGraphDestroy(decode_graph_);
}

int QwenModel::prefill(const CPUTensor &c_input_i32) {
    // 为一次新生成重建 session（丢弃上一次请求的 KV cache / recurrent state）。
    session_ = std::make_unique<QwenSession>(config_, c_input_i32, max_output_tokens_);
    // 新 session 的 device buffer 地址变了，旧 graph 失效，下次贪心 decode 需重新预热 + 捕获。
    decode_graph_ready_ = false;
    decode_graph_session_ = nullptr;
    decode_greedy_steps_ = 0;
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
    // 非贪心（温度/top-k/top-p/重复惩罚）需 host 端 Sampler，无法走 GPU argmax 闭环，
    // 退回原逐 kernel launch 路径。
    if (!sampler_.is_greedy()) {
        const TextConfig &text_config = config_.data.text;
        const int hidden_size = text_config.hidden_size;
        CudaScratch &scratch = session.scratch;

        GPUTensor g_hidden_f32 = GPUTensor(
            scratch, scratch_key::kHidden, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
        const int token_id = prev_token_id;
        CPUTensor c_input_view_i32 = CPUTensor(&token_id, {1}, DType::I32);
        embedding_.forward(weights_.s_token_embd, c_input_view_i32, g_hidden_f32, scratch);
        for (DecoderLayer &layer : layers_) layer.decode(session, g_hidden_f32, pos);
        GPUTensor g_normed_f32 = GPUTensor(
            scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
        RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                         config_.data.text.rms_norm_eps, /*one_plus=*/true);
        return lm_head_.forward(weights_.s_token_embd, session, g_normed_f32, sampler_);
    }

    // === 贪心：CUDA Graph 路径 ===
    cudaStream_t stream = get_current_cuda_stream();
    // 每步 graph 外把 host 侧的 pos / 输入 token 异步写入 device buffer（graph 内 kernel 只读它们）。
    check_cuda(cudaMemcpyAsync(session.d_pos(), &pos, sizeof(int), cudaMemcpyHostToDevice, stream),
               "decode pos H2D 失败");
    check_cuda(cudaMemcpyAsync(session.d_token(), &prev_token_id, sizeof(int), cudaMemcpyHostToDevice, stream),
               "decode token H2D 失败");

    // profile 采样步走 eager：graph replay 内的 kernel 不经 ScopedGpuTimer 埋点，
    // 无法拿到逐 kernel 细分；此步改走 eager 让 profiler 抓到每个 kernel 的耗时。
    // 非采样步仍走 graph，保持稳态吞吐口径不变。
    const bool profile_this_step = Profiler::instance().capturing();

    if (decode_greedy_steps_ == 0 || decode_graph_session_ != &session || profile_this_step) {
        // 首步（或换 session / profile 采样步）走 eager：把所有 grow-only scratch 撑到
        // decode 稳态尺寸，避免随后 capture 期间触发非法的 cudaMalloc；同时 GPU argmax 把结果写回 d_token。
        eager_decode_greedy_device(session);
        // profile 采样步不使已建好的 graph 失效（它没改 scratch 尺寸，下步可继续 replay）。
        if (!profile_this_step) {
            decode_graph_ready_ = false;      // scratch 刚可能增长，之前若有 graph 也失效
            decode_graph_session_ = &session; // 记录当前 session（预热已针对它）
        }
    } else {
        if (!decode_graph_ready_) {
            record_decode_graph(session);
        }
        check_cuda(cudaGraphLaunch(decode_graph_exec_, stream), "decode graph launch 失败");
    }
    ++decode_greedy_steps_;

    // 取回 argmax 得到的下一个 token id（graph 外的一次 int D2H + 同流同步）。
    int next_token = 0;
    cuda_memcpy_d2h(&next_token, session.d_token(), sizeof(int), "decode token D2H 失败");
    return next_token;
}

void QwenModel::eager_decode_greedy_device(QwenSession &session) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    CudaScratch &scratch = session.scratch;
    cudaStream_t stream = get_current_cuda_stream();

    GPUTensor g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    TensorTool::embedding_lookup_device(weights_.s_token_embd, session.d_token(), g_hidden_f32, stream);
    for (DecoderLayer &layer : layers_) layer.decode(session, g_hidden_f32, /*pos=*/0);
    GPUTensor g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);
    lm_head_.forward_argmax_device(weights_.s_token_embd, session, g_normed_f32, session.d_token(), stream);
}

void QwenModel::record_decode_graph(QwenSession &session) {
    const TextConfig &text_config = config_.data.text;
    const int hidden_size = text_config.hidden_size;
    CudaScratch &scratch = session.scratch;
    cudaStream_t stream = get_current_cuda_stream();

    // 若之前有旧 graph（换 session 重建），先销毁。
    if (decode_graph_exec_) {
        cudaGraphExecDestroy(decode_graph_exec_);
        decode_graph_exec_ = nullptr;
    }
    if (decode_graph_) {
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
    }

    // ThreadLocal 模式：仅捕获本线程当前流上的操作，避免误捕获库内部的其它流活动。
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "decode graph begin capture 失败");

    GPUTensor g_hidden_f32 = GPUTensor(
        scratch, scratch_key::kHidden, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    // embedding 从 device token buffer 读输入 token（不做 H2D）。
    TensorTool::embedding_lookup_device(weights_.s_token_embd, session.d_token(), g_hidden_f32, stream);

    for (DecoderLayer &layer : layers_) {
        // pos 走 session.d_pos()（device），此处 host pos 仅用于层内已废弃的 seq_len 记账，传 0 即可。
        layer.decode(session, g_hidden_f32, /*pos=*/0);
    }

    GPUTensor g_normed_f32 = GPUTensor(
        scratch, scratch_key::kTokenHiddenA, {1, static_cast<int64_t>(hidden_size)}, DType::F32);
    RMSNorm::forward(weights_.s_output_norm, g_hidden_f32, g_normed_f32,
                     config_.data.text.rms_norm_eps, /*one_plus=*/true);
    // lm_head GEMM + GPU argmax，把下一 token 写回 device token buffer（闭环，无 D2H）。
    lm_head_.forward_argmax_device(weights_.s_token_embd, session, g_normed_f32, session.d_token(), stream);

    check_cuda(cudaStreamEndCapture(stream, &decode_graph_), "decode graph end capture 失败");
    check_cuda(cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0),
               "decode graph instantiate 失败");

    decode_graph_ready_ = true;
    decode_graph_session_ = &session;
}
