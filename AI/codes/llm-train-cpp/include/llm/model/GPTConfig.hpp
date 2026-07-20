#pragma once

#include "llm/core.hpp"

namespace llm {

// GPT 模型配置。
// 这里保存模型结构、dropout、设备等超参数。
struct GPTConfig {
    // 词表大小，GPT-2 BPE 默认是 50257。
    int64_t vocab_size{50257};

    // 最大上下文长度，也就是一次能看的 token 数。
    int64_t context_length{256};

    // token embedding 维度。
    int64_t emb_dim{768};

    // 多头注意力的 head 数。
    int64_t n_heads{12};

    // TransformerBlock 层数。
    int64_t n_layers{12};

    // dropout 概率；当前简化实现中可能不会完整使用。
    double drop_rate{0.1};

    // Q/K/V 线性投影是否使用 bias。
    bool qkv_bias{false};

    // 模型参数和计算所在设备。
    Device device{};
};

} // namespace llm
