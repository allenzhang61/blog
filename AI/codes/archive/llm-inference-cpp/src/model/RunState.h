#pragma once

#include "../core/config.h"

#include <cstddef>
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
class RunState {
public:
    // 按模型配置创建一次生成所需的运行状态。
    RunState(const ModelConfig & config, int max_seq_len);
    RunState(const RunState &) = delete;
    RunState & operator=(const RunState &) = delete;
    RunState(RunState &&) noexcept = default;
    RunState & operator=(RunState &&) noexcept = default;

    // 返回已经处理过的 token 数量。
    int sequence_length() const noexcept {
        return sequence_length_;
    }

    // 返回模型层数。
    std::size_t layer_count() const noexcept {
        return linear_states_.size();
    }

    // 访问指定层的 linear attention 运行状态。
    LinearAttentionState & linear_state(std::size_t layer) noexcept {
        return linear_states_[layer];
    }
    const LinearAttentionState & linear_state(std::size_t layer) const noexcept {
        return linear_states_[layer];
    }

    // 访问指定层的 full attention 运行状态。
    FullAttentionState & full_state(std::size_t layer) noexcept {
        return full_states_[layer];
    }
    const FullAttentionState & full_state(std::size_t layer) const noexcept {
        return full_states_[layer];
    }

    // 记录本次新处理的 token 数量。
    void advance(int token_count) noexcept {
        sequence_length_ += token_count;
    }

private:
    // 已处理的序列长度。
    int sequence_length_ = 0;
    // 每层 linear attention 状态。
    std::vector<LinearAttentionState> linear_states_;
    // 每层 full attention 状态。
    std::vector<FullAttentionState> full_states_;
};

} // namespace llm_inference
