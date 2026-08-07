#include "runtime_state.h"

#include "../kernels/cuda/cuda_ops.h"

namespace llm_inference {

LinearAttentionState::LinearAttentionState(LinearAttentionState && other) noexcept
    : cuda_state(other.cuda_state) {
    other.cuda_state = nullptr;
}

LinearAttentionState & LinearAttentionState::operator=(LinearAttentionState && other) noexcept {
    if (this != &other) {
        cuda_free_linear_attention_state(cuda_state);
        cuda_state = other.cuda_state;
        other.cuda_state = nullptr;
    }
    return *this;
}

LinearAttentionState::~LinearAttentionState() {
    cuda_free_linear_attention_state(cuda_state);
}

FullAttentionState::FullAttentionState(FullAttentionState && other) noexcept
    : max_seq_len(other.max_seq_len),
      cuda_state(other.cuda_state) {
    other.cuda_state = nullptr;
}

FullAttentionState & FullAttentionState::operator=(FullAttentionState && other) noexcept {
    if (this != &other) {
        cuda_free_full_attention_state(cuda_state);
        max_seq_len = other.max_seq_len;
        cuda_state = other.cuda_state;
        other.cuda_state = nullptr;
    }
    return *this;
}

FullAttentionState::~FullAttentionState() {
    cuda_free_full_attention_state(cuda_state);
}

RunState make_run_state(const ModelConfig & config, int max_seq_len) {
    RunState state;
    state.linear.resize(config.text.num_hidden_layers);
    state.full.resize(config.text.num_hidden_layers);
    for (int layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        if (config.text.layer_types[layer] != "linear_attention") {
            state.full[layer].max_seq_len = max_seq_len;
        }
    }
    return state;
}

} // namespace llm_inference

