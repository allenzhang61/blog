#include "QwenDecoder.h"

#include <cstddef>
#include <stdexcept>

namespace llm_inference {

QwenDecoder::QwenDecoder(const ModelConfig & config, const ModelParams & params)
    : embedding_(params.embed_tokens), prefill_runner_(config, params) {
    blocks_.reserve(params.layers.size());
    for (size_t i = 0; i < params.layers.size(); ++i) {
        const LayerWeights & layer = params.layers[i];
        if (layer.type == "linear_attention") {
            blocks_.push_back(std::make_unique<QwenLinearAttentionBlock>(config, layer, static_cast<int>(i)));
        } else if (layer.type == "full_attention") {
            blocks_.push_back(std::make_unique<QwenFullAttentionBlock>(config, layer, static_cast<int>(i)));
        } else {
            throw std::runtime_error("未知 Qwen layer type：" + layer.type);
        }
    }
}

const char * QwenDecoder::name() const {
    return "QwenDecoder";
}

Tensor QwenDecoder::prefill(const std::vector<int> & input_ids, RunState & state) const {
    return prefill_runner_.forward(input_ids, state);
}

Tensor QwenDecoder::forward(const Tensor & device_token_id, RunState & state) const {
    Tensor current = embedding_.forward(device_token_id);

    for (const auto & block : blocks_) {
        current = block->forward(current, state);
    }

    state.advance(1);
    return current;
}

} // namespace llm_inference
