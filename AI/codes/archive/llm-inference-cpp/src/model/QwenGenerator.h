#pragma once

#include "../core/cli.h"
#include "../core/config.h"
#include "../core/profile.h"
#include "QwenModel.h"
#include "RunState.h"

#include <vector>

namespace llm_inference {

// Greedy decode 生成器，负责 prefill/decode 编排、计时和输出 token 管理。
class QwenGenerator {
public:
    // 绑定只读模型和配置，不拥有生命周期。
    QwenGenerator(const QwenModel & model, const ModelConfig & config);

    // 使用 CUDA greedy 路径生成完整序列。
    std::vector<int> generate(
        RunState & state,
        const Args & args,
        const std::vector<int> & input_ids,
        Timing & timing) const;

private:
    const QwenModel & model_;
    const ModelConfig & config_;
};

} // namespace llm_inference
