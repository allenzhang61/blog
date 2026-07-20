#pragma once

#include "llm/model/GELU.hpp"
#include "llm/model/GPTConfig.hpp"
#include "llm/model/Linear.hpp"

namespace llm {

// Transformer 中的前馈网络。
// 典型结构是 Linear: emb_dim -> 4 * emb_dim，GELU，再 Linear 回 emb_dim。
class FeedForward : public Module {
public:
    // 第一层线性变换，扩展隐藏维度。
    Linear fc1;

    // 非线性激活。
    GELU gelu;

    // 第二层线性变换，压回 embedding 维度。
    Linear fc2;

    // 按 GPTConfig 初始化前馈网络。
    explicit FeedForward(const GPTConfig& cfg);

    // 输入和输出形状相同，通常是 [batch, seq_len, emb_dim]。
    Tensor forward(const Tensor& x);

    // 汇总 fc1 和 fc2 的可训练参数。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
