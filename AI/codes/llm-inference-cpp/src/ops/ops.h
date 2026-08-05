#pragma once

#include "../core/config.h"
#include "../model/weights.h"

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

// ops 分发层：集中「try-CUDA-then-CPU」逻辑，取代散落在模型层的手写 fallback。
// CUDA 多级融合（整层 / project / 裸算子）在各 ops 入口内部消化。
namespace ops {

// 整层融合：input_norm + linear attention + post_norm + MLP（主机输入/输出）。
// 成功走 CUDA 返回 true；否则返回 false，由调用方走更细粒度路径。
bool linear_attention_full_layer(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearLayerState & state,
    std::vector<float> & out);

// 整层融合：input_norm + full attention + post_norm + MLP（主机输入/输出）。
bool full_attention_full_layer(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out);

// RMSNorm + linear attention projection 融合。成功返回 true。
bool rmsnorm_linear_attention_project(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearLayerState & state,
    std::vector<float> & out);

// RMSNorm + full attention projection 融合。成功返回 true。
bool rmsnorm_full_attention_project(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out);

// RMSNorm + MLP 融合（仅当环境启用时尝试）。成功返回 true。
bool rmsnorm_mlp(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    std::vector<float> & out);

// mixer：linear attention（try CUDA project/core，否则 CPU）。已在 input_norm 之后。
void linear_attention(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearLayerState & state,
    std::vector<float> & out);

// mixer：full attention（try CUDA project/core，否则 CPU）。已在 input_norm 之后。
void full_attention(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out);

// MLP：try CUDA fused，否则 CPU（gate/up/down + silu）。已在 post_norm 之后。
void mlp(
    const LayerWeights & w,
    const std::vector<float> & x,
    std::vector<float> & out);

// 从主机 hidden 计算 logits argmax（try CUDA，否则 CPU）。
int argmax_logits(const ModelParams & params, const std::vector<float> & hidden);

} // namespace ops

} // namespace llm_inference
