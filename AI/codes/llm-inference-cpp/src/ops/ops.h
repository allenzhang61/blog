#pragma once

#include "../core/config.h"
#include "../model/weights.h"

#include <vector>

namespace llm_inference {

// linear attention 层的运行时缓存。
struct LinearAttentionState {
    // depthwise conv 的滑动窗口状态。
    std::vector<float> conv_state;
    // recurrent linear attention 状态。
    std::vector<float> recurrent_state;
    // CUDA 侧 fused linear attention state。
    void * cuda_state = nullptr;

    LinearAttentionState() = default;
    LinearAttentionState(const LinearAttentionState &) = delete;
    LinearAttentionState & operator=(const LinearAttentionState &) = delete;
    LinearAttentionState(LinearAttentionState && other) noexcept;
    LinearAttentionState & operator=(LinearAttentionState && other) noexcept;
    ~LinearAttentionState();
};

// full attention 层的运行时 KV cache。
struct FullAttentionState {
    // CPU 路径使用的 key cache。
    std::vector<float> key_cache;
    // CPU 路径使用的 value cache。
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

// ops CPU 后端：严格设备匹配下 Device::CPU 的算子实现。
// CUDA 路径完全由设备端 forward（QwenModel::forward_token_device_*）承担，此处不做任何回退。
namespace ops {

// CPU 矩阵向量乘（BLAS，退化时手写循环）。
void matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y);

// CPU mixer：linear attention。已在 input_norm 之后。
void linear_attention(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearAttentionState & state,
    std::vector<float> & out);

// CPU mixer：full attention。已在 input_norm 之后。
void full_attention(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out);

// CPU MLP（gate/up/down + silu）。已在 post_norm 之后。
void mlp(
    const LayerWeights & w,
    const std::vector<float> & x,
    std::vector<float> & out);

// CPU：从主机 hidden 计算 logits argmax。
int argmax_logits(const ModelParams & params, const std::vector<float> & hidden);

} // namespace ops

} // namespace llm_inference
