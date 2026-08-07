#pragma once

#include "../core/config.h"

#include <vector>

namespace llm_inference {

// linear attention 层的运行时 CUDA cache。
struct LinearAttentionState {
    void * cuda_state = nullptr;

    LinearAttentionState() = default;
    LinearAttentionState(const LinearAttentionState &) = delete;
    LinearAttentionState & operator=(const LinearAttentionState &) = delete;
    LinearAttentionState(LinearAttentionState && other) noexcept;
    LinearAttentionState & operator=(LinearAttentionState && other) noexcept;
    ~LinearAttentionState();
};

// full attention 层的运行时 CUDA KV cache。
struct FullAttentionState {
    int max_seq_len = 0;
    void * cuda_state = nullptr;

    FullAttentionState() = default;
    FullAttentionState(const FullAttentionState &) = delete;
    FullAttentionState & operator=(const FullAttentionState &) = delete;
    FullAttentionState(FullAttentionState && other) noexcept;
    FullAttentionState & operator=(FullAttentionState && other) noexcept;
    ~FullAttentionState();
};

// 单次生成过程的跨 token 运行状态。
struct RunState {
    // 已处理的序列长度。
    int seq_len = 0;
    // 每层 linear attention 状态。
    std::vector<LinearAttentionState> linear;
    // 每层 full attention 状态。
    std::vector<FullAttentionState> full;
};

// 按模型配置初始化一次生成所需的运行状态。
RunState make_run_state(const ModelConfig & config, int max_seq_len);

} // namespace llm_inference

