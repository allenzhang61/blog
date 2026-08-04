#pragma once

#include "../core/cli.h"
#include "../core/config.h"
#include "../core/profile.h"
#include "../core/safetensors.h"
#include "../core/tensor_ops.h"

#include <string>
#include <vector>

namespace llm_inference {

// linear attention 层的运行时缓存。
struct LinearLayerState {
    // depthwise conv 的滑动窗口状态。
    std::vector<float> conv_state;
    // recurrent linear attention 状态。
    std::vector<float> recurrent_state;
    // CUDA 侧 fused linear attention state。
    void * cuda_state = nullptr;

    LinearLayerState() = default;
    LinearLayerState(const LinearLayerState &) = delete;
    LinearLayerState & operator=(const LinearLayerState &) = delete;
    LinearLayerState(LinearLayerState && other) noexcept;
    LinearLayerState & operator=(LinearLayerState && other) noexcept;
    ~LinearLayerState();
};

// full attention 层的运行时 KV cache。
struct FullAttentionState {
    // CPU fallback 使用的 key cache。
    std::vector<float> key_cache;
    // CPU fallback 使用的 value cache。
    std::vector<float> value_cache;
    // 当前层 KV cache 支持的最大序列长度。
    int max_seq_len = 0;
    // CUDA 侧 fused full attention state。
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
    std::vector<LinearLayerState> linear;
    // 每层 full attention 状态。
    std::vector<FullAttentionState> full;
};

// 校验 Qwen3.5 推理路径需要的 tensor 是否齐全。
void validate_qwen_tensors(const ModelWeights & weights, const ModelConfig & config);

// 按模型配置初始化一次生成所需的运行状态。
RunState make_run_state(const ModelConfig & config, int max_seq_len);

// Qwen3.5 原生推理实现，封装单 token forward 和生成循环。
class NativeQwen {
public:
    // 绑定只读模型配置和 mmap 权重，不拥有二者生命周期。
    NativeQwen(const ModelConfig & config, const ModelWeights & weights);

    // 对 prompt 做 prefill，并生成第一个 token。
    int generate_next(const std::vector<int> & prompt_ids, RunState & state, Timing & timing) const;

    // 基于上一个 token decode 一个新 token。
    int decode_one(int token, RunState & state, Timing & timing) const;

    // 使用 CUDA 批量 prefill + 设备端 decode 生成完整序列。
    bool generate_sequence_device(
        const std::vector<int> & prompt_ids,
        RunState & state,
        int max_new_tokens,
        int eos_token_id,
        Timing & timing,
        std::vector<int> & generated) const;

private:
    // 按名称获取权重 tensor。
    TensorRef t(const std::string & name) const;

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

    // MLP 层 CPU/CUDA 路径封装。
    void mlp_layer(const std::string & prefix, const std::vector<float> & x, std::vector<float> & out) const;

    // linear attention 层 CPU/CUDA 路径封装。
    void linear_attention_layer(
        const std::string & prefix,
        const std::vector<float> & x,
        LinearLayerState & state,
        std::vector<float> & out) const;

    // 对单个 full attention head 应用 RoPE。
    void apply_rope(float * vec, int pos) const;

    // full attention 层 CPU/CUDA 路径封装。
    void full_attention_layer(
        const std::string & prefix,
        const std::vector<float> & x,
        FullAttentionState & state,
        int pos,
        std::vector<float> & out) const;

    // CPU logits argmax。
    int argmax_logits(const std::vector<float> & hidden, Timing & timing) const;

    // CUDA final norm + logits argmax。
    bool argmax_logits_device(const void * device_hidden, Timing & timing, int & best_id) const;

    // 模型结构配置引用。
    const ModelConfig & config_;
    // mmap 权重引用。
    const ModelWeights & weights_;
};

// 基于输入 token ids 执行完整生成流程，返回新生成的 token ids。
std::vector<int> run_generation(
    const ModelConfig & config,
    const ModelWeights & weights,
    const Args & args,
    const std::vector<int> & input_ids,
    Timing & timing);

} // namespace llm_inference
