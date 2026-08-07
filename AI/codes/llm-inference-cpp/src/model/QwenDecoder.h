#pragma once

#include "Embedding.h"
#include "Tensor.h"
#include "Module.h"
#include "QwenFullAttentionBlock.h"
#include "QwenBlock.h"
#include "QwenLinearAttentionBlock.h"
#include "runtime_state.h"
#include "weights.h"
#include "../core/config.h"

#include <memory>
#include <vector>

namespace llm_inference {

// Qwen3.5 的 Transformer block 堆叠，负责 prefill 和 decode 层间流转。
class QwenDecoder : public Module {
public:
    QwenDecoder(const ModelConfig & config, const ModelParams & params);

    const char * name() const override;

    // batch prefill 完整 prompt，返回最后一个 prompt token 的 device hidden。
    Tensor prefill(const std::vector<int> & input_ids, RunState & state) const;

    // decode 阶段从一个 token id 前进一轮，返回下一轮 device hidden。
    Tensor forward(const Tensor & device_token_id, RunState & state) const;

private:
    const ModelConfig & config_;
    const ModelParams & params_;
    TokenEmbedding embedding_;
    std::vector<std::unique_ptr<QwenBlock>> blocks_;
};

} // namespace llm_inference
