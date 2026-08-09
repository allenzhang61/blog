#pragma once

#include "Tensor.h"
#include "Module.h"
#include "../safetensors/safetensors.h"

namespace llm_inference {

// Token embedding 模块，把 token id 映射成 device hidden。
class TokenEmbedding : public Module {
public:
    explicit TokenEmbedding(const WeightData & weight);

    const char * name() const override;

    // 从 device token id 做 embedding lookup，返回 device hidden。
    Tensor forward(const Tensor & device_token_id) const;

private:
    WeightData weight_;
};

} // namespace llm_inference
