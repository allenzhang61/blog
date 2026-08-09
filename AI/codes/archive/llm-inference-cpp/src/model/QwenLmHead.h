#pragma once

#include "Tensor.h"
#include "Module.h"
#include "../safetensors/safetensors.h"

namespace llm_inference {

// 最后一层 RMSNorm + tied lm_head + greedy argmax。
class QwenLmHead : public Module {
public:
    QwenLmHead(const WeightData & final_norm, const WeightData & embedding, float rms_norm_eps);

    const char * name() const override;

    // 根据当前 hidden 选出下一个 token id，并保留在 device_token_out 中。
    void forward(const Tensor & device_hidden, const Tensor & device_token_out) const;

private:
    WeightData final_norm_;
    WeightData embedding_;
    int hidden_size_ = 0;
    float rms_norm_eps_ = 0.0f;
};

} // namespace llm_inference
