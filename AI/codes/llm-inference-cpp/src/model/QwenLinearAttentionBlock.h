#pragma once

#include "QwenBlock.h"
#include "QwenLinearAttention.h"
#include "QwenMlp.h"

namespace llm_inference {

// linear attention 类型的 Qwen transformer 层。
class QwenLinearAttentionBlock : public QwenBlock {
public:
    QwenLinearAttentionBlock(const ModelConfig & config, const LayerWeights & weights, int layer_index);

    const char * name() const override;
    Tensor forward(const Tensor & device_x, RunState & state) const override;

private:
    QwenLinearAttention linear_attention_;
    QwenMlp mlp_;
};

} // namespace llm_inference
