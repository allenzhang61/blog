#pragma once

#include "QwenBlock.h"
#include "QwenFullAttention.h"
#include "QwenMlp.h"

namespace llm_inference {

// full attention 类型的 Qwen transformer 层。
class QwenFullAttentionBlock : public QwenBlock {
public:
    QwenFullAttentionBlock(const ModelConfig & config, const LayerWeights & weights, int layer_index);

    const char * name() const override;
    Tensor forward(const Tensor & device_x, RunState & state) const override;

private:
    QwenFullAttention full_attention_;
    QwenMlp mlp_;
};

} // namespace llm_inference
