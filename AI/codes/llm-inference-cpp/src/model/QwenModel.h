#pragma once

#include "../core/cli.h"
#include "../core/config.h"
#include "../core/profile.h"
#include "../safetensors/safetensors.h"
#include "weights.h"

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

// Qwen3.5 原生推理实现，封装单 token forward 和生成循环。
// 持有一次性解析好的结构化权重引用（ModelParams）。
class QwenModel {
public:
    // 绑定只读模型配置和 mmap 权重，构造时解析权重引用，不拥有生命周期。
    QwenModel(const ModelConfig & config, const ModelWeights & weights);

    // 使用 CUDA greedy 路径生成完整序列。
    std::vector<int> generate(
        RunState & state,
        const Args & args,
        const std::vector<int> & input_ids,
        Timing & timing) const;

private:
    // 从主机或设备端 token id 开始执行设备端 forward；失败抛异常。
    const void * forward_token_device(int token, const void * device_token, RunState & state) const;

    // 执行所有 transformer 层的设备端 forward。
    const void * forward_token_device_layers(void * current, void * next, RunState & state) const;

    // 模型结构配置引用。
    const ModelConfig & config_;
    // 一次性解析好的结构化权重引用。
    ModelParams params_;
};

} // namespace llm_inference
