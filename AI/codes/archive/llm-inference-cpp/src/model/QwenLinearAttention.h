#pragma once

#include "Module.h"
#include "Tensor.h"
#include "RunState.h"
#include "weights.h"
#include "../core/config.h"

namespace llm_inference {

// Qwen linear attention mixer 子模块，不包含 block 外层 RMSNorm/MLP/residual。
class QwenLinearAttention : public Module {
public:
    QwenLinearAttention(const ModelConfig & config, const LinearAttnWeights & weights, int layer_index);

    const char * name() const override;

    // 输入为 input RMSNorm 后的 device 低精度 hidden，输出为 attention mixer float hidden。
    Tensor forward(const Tensor & normed_x, RunState & state) const;

private:
    const ModelConfig & config_;
    const LinearAttnWeights & weights_;
    int layer_index_ = 0;
};

} // namespace llm_inference
