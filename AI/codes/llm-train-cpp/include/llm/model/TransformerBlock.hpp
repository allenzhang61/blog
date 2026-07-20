#pragma once

#include "llm/model/FeedForward.hpp"
#include "llm/model/LayerNorm.hpp"
#include "llm/model/MultiHeadAttention.hpp"

namespace llm {

// 一个 GPT Transformer Block。
// 结构是 Pre-LN Attention + 残差，再 Pre-LN FeedForward + 残差。
class TransformerBlock : public Module {
public:
    // 多头因果自注意力子层。
    MultiHeadAttention att;

    // 前馈网络子层。
    FeedForward ff;

    // 注意力前的 LayerNorm。
    LayerNorm norm1;

    // 前馈网络前的 LayerNorm。
    LayerNorm norm2;

    // 按 GPTConfig 初始化一个 Transformer block。
    explicit TransformerBlock(const GPTConfig& cfg);

    // 输入和输出形状相同，通常是 [batch, seq_len, emb_dim]。
    Tensor forward(const Tensor& x);

    // 汇总 attention、feed-forward、layernorm 的参数。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
