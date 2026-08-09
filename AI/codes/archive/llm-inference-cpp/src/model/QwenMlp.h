#pragma once

#include "Module.h"
#include "Tensor.h"
#include "weights.h"

namespace llm_inference {

// Qwen MLP 子模块：gate/up projection -> SiLU(gate) * up -> down projection。
class QwenMlp : public Module {
public:
    QwenMlp(const MlpWeights & weights, int layer_index);

    const char * name() const override;

    // 输入为 device 低精度 hidden，输出为 float device hidden。
    Tensor forward(const Tensor & device_x) const;

private:
    const MlpWeights & weights_;
    int layer_index_ = 0;
};

} // namespace llm_inference

