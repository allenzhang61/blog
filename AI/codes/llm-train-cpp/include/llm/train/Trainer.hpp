#pragma once

#include "llm/data/DataLoader.hpp"
#include "llm/model/GPTModel.hpp"
#include "llm/train/AdamW.hpp"

namespace llm {

// 训练和生成的辅助工具。
// 这里放静态方法，避免把训练循环塞进 GPTModel 本身。
class Trainer {
public:
    // 遍历 DataLoader 一轮，并返回平均 loss。
    static double train_one_epoch(GPTModel& model, DataLoader& loader, AdamW& optim);

    // 训练固定步数，并返回最后一次或平均 loss。
    static double train_steps(GPTModel& model, DataLoader& loader, AdamW& optim, int64_t steps);

    // 贪心生成：每一步取 logits 最大的 token 追加到序列后。
    static std::vector<int64_t> generate_greedy(GPTModel& model, std::vector<int64_t> ids,
                                                int64_t max_new_tokens, int64_t context);
};

} // namespace llm
