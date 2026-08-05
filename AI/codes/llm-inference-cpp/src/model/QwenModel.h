#pragma once

#include "../core/cli.h"
#include "../core/config.h"
#include "../core/profile.h"
#include "../ops/ops.h"
#include "../safetensors/safetensors.h"
#include "weights.h"

#include <vector>

namespace llm_inference {

// 单次生成过程的跨 token 运行状态。
struct RunState {
    // 已处理的序列长度。
    int seq_len = 0;
    // 每层 linear attention 状态。
    std::vector<LinearLayerState> linear;
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

    // 使用 greedy 路径生成完整序列，优先走整段 CUDA，失败后回退到逐 token。
    std::vector<int> run_greedy_generation(
        RunState & state,
        const Args & args,
        const std::vector<int> & input_ids,
        Timing & timing) const;

    // 使用非 greedy 路径生成完整序列；当前实现仍复用逐 token greedy。
    std::vector<int> run_non_greedy_generation(
        RunState & state,
        const Args & args,
        const std::vector<int> & input_ids,
        Timing & timing) const;

    // 使用逐 token 路径生成完整序列。
    std::vector<int> run_token_generation(
        RunState & state,
        const Args & args,
        const std::vector<int> & input_ids,
        Timing & timing) const;

    // 使用 CUDA 批量 prefill + 设备端 decode 生成完整序列。
    bool generate_sequence_device(
        const std::vector<int> & prompt_ids,
        RunState & state,
        int max_new_tokens,
        int eos_token_id,
        Timing & timing,
        std::vector<int> & generated) const;

private:
    // 对 prompt 做 prefill，并生成第一个 token。
    int generate_next(const std::vector<int> & prompt_ids, RunState & state, Timing & timing) const;

    // 基于上一个 token decode 一个新 token。
    int decode_one(int token, RunState & state, Timing & timing) const;

    // CUDA 单 token prefill 后取 argmax。
    bool generate_next_device(const std::vector<int> & prompt_ids, RunState & state, Timing & timing, int & next) const;

    // CUDA decode 一个 token 后取 argmax。
    bool decode_one_device(int token, RunState & state, Timing & timing, int & next) const;

    // 从主机 token id 开始执行设备端 forward。
    const void * forward_token_device(int token, RunState & state) const;

    // 从设备端 token id 开始执行设备端 forward。
    const void * forward_token_device_from_device(const void * device_token, RunState & state) const;

    // 执行所有 transformer 层的设备端 forward。
    const void * forward_token_device_layers(void * current, void * next, RunState & state) const;

    // CPU fallback 单 token forward，返回 final norm 后 hidden。
    std::vector<float> forward_token(int token, RunState & state) const;

    // CPU logits argmax（含计时）。
    int argmax_logits(const std::vector<float> & hidden, Timing & timing) const;

    // CUDA final norm + logits argmax。
    bool argmax_logits_device(const void * device_hidden, Timing & timing, int & best_id) const;

    // 模型结构配置引用。
    const ModelConfig & config_;
    // mmap 权重引用。
    const ModelWeights & weights_;
    // 一次性解析好的结构化权重引用。
    ModelParams params_;
};

} // namespace llm_inference
