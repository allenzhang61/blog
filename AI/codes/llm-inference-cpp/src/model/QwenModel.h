#pragma once

#include "../core/config.h"
#include "Tensor.h"
#include "Module.h"
#include "QwenDecoder.h"
#include "QwenLmHead.h"
#include "runtime_state.h"
#include "weights.h"

#include <vector>

namespace llm_inference {

// Qwen3.5 原生模型结构，封装 prefill、单 token forward 和 lm head。
// 持有 embedding、decoder、lm head 等子模块。
class QwenModel : public Module {
public:
    // 绑定只读模型配置和 mmap 权重，构造时解析权重引用，不拥有生命周期。
    QwenModel(const ModelConfig & config, const ModelWeights & weights);

    const char * name() const override;

    // batch prefill 完整 prompt，返回最后一个 prompt token 的 device hidden。
    Tensor prefill(const std::vector<int> & input_ids, RunState & state) const;

    // decode 阶段执行一次完整模型 forward，输入 token id 在 device 上，返回下一轮 hidden。
    Tensor forward(const Tensor & device_token_id, RunState & state) const;

    // 根据当前 hidden 选出下一个 token id，并写入指定的 device token slot。
    void forward_lm_head(const Tensor & device_hidden, const Tensor & device_token_out) const;

private:
    // 模型结构配置引用。
    const ModelConfig & config_;
    // 一次性解析好的结构化权重引用。
    ModelParams params_;
    // Transformer block 堆叠，负责 prefill 和 decode 单 token forward。
    QwenDecoder decoder_;
    // tied lm head，负责 final norm + logits argmax。
    QwenLmHead lm_head_;
};

} // namespace llm_inference
