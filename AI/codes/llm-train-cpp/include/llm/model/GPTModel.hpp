#pragma once

#include "llm/model/Embedding.hpp"
#include "llm/model/LayerNorm.hpp"
#include "llm/model/Linear.hpp"
#include "llm/model/TransformerBlock.hpp"

namespace llm {

// 简化版 GPT 语言模型。
// 输入 token id，输出每个位置预测下一个 token 的 logits。
class GPTModel : public Module {
public:
    // 模型配置。
    GPTConfig cfg;

    // token embedding，把词表 id 映射成向量。
    Embedding tok_emb;

    // position embedding，为每个位置提供位置信息。
    Embedding pos_emb;

    // 堆叠的 Transformer blocks。
    std::vector<TransformerBlock> blocks;

    // 最后一层归一化。
    LayerNorm final_norm;

    // 输出头，把 embedding 投影回 vocab_size。
    Linear out_head;

    // 使用配置初始化完整 GPT 模型。
    explicit GPTModel(GPTConfig cfg_);

    // ids 形状通常是 [batch, seq_len]，输出 logits 是 [batch, seq_len, vocab_size]。
    Tensor forward(const Tensor& ids);

    // 汇总模型所有可训练参数。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
