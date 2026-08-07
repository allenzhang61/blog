#pragma once

#include "Module.h"
#include "Tensor.h"
#include "runtime_state.h"
#include "weights.h"
#include "../core/config.h"

namespace llm_inference {

// Qwen3.5 单个 Transformer 层的抽象基类。
class QwenBlock : public Module {
public:
    QwenBlock(const ModelConfig & config, const LayerWeights & weights, int layer_index);
    ~QwenBlock() override = default;

    // decode 阶段的 batch=1 整层 forward，输入/输出都在 device 上。
    virtual Tensor forward(const Tensor & device_x, RunState & state) const = 0;

protected:
    // 为当前层分配输出 hidden buffer。
    Tensor allocate_output(const Tensor & device_x) const;

    const ModelConfig & config_;
    const LayerWeights & weights_;
    int layer_index_ = 0;
};

} // namespace llm_inference
