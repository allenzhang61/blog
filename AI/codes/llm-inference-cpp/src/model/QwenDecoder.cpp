#include "QwenDecoder.h"

#include "../kernels/cuda/cuda_ops.h"

#include <stdexcept>

namespace llm_inference {

QwenDecoder::QwenDecoder(const ModelConfig & config, const ModelParams & params)
    : config_(config), params_(params), embedding_(params.embed_tokens) {
    blocks_.reserve(params_.layers.size());
    for (size_t i = 0; i < params_.layers.size(); ++i) {
        const LayerWeights & layer = params_.layers[i];
        if (layer.type == "linear_attention") {
            blocks_.push_back(std::make_unique<QwenLinearAttentionBlock>(config_, layer, static_cast<int>(i)));
        } else if (layer.type == "full_attention") {
            blocks_.push_back(std::make_unique<QwenFullAttentionBlock>(config_, layer, static_cast<int>(i)));
        } else {
            throw std::runtime_error("未知 Qwen layer type：" + layer.type);
        }
    }
}

const char * QwenDecoder::name() const {
    return "QwenDecoder";
}

Tensor QwenDecoder::prefill(const std::vector<int> & input_ids, RunState & state) const {
    std::vector<void *> linear_cuda_states(state.linear.size(), nullptr);
    std::vector<void *> full_cuda_states(state.full.size(), nullptr);
    std::vector<int> full_max_seq_lens(state.full.size(), 0);
    for (size_t i = 0; i < state.linear.size(); ++i) {
        linear_cuda_states[i] = state.linear[i].cuda_state;
    }
    for (size_t i = 0; i < state.full.size(); ++i) {
        full_cuda_states[i] = state.full[i].cuda_state;
        full_max_seq_lens[i] = state.full[i].max_seq_len;
    }

    const void * hidden = cuda_prefill_batch(
        config_,
        params_,
        input_ids,
        linear_cuda_states,
        full_cuda_states,
        full_max_seq_lens,
        state.seq_len);

    for (size_t i = 0; i < state.linear.size(); ++i) {
        state.linear[i].cuda_state = linear_cuda_states[i];
    }
    for (size_t i = 0; i < state.full.size(); ++i) {
        state.full[i].cuda_state = full_cuda_states[i];
    }
    return {const_cast<void *>(hidden), config_.text.hidden_size, -1};
}

Tensor QwenDecoder::forward(const Tensor & device_token_id, RunState & state) const {
    Tensor current = embedding_.forward(device_token_id);

    for (const auto & block : blocks_) {
        current = block->forward(current, state);
    }

    state.seq_len += 1;
    return current;
}

} // namespace llm_inference
