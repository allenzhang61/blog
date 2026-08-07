#include "QwenModel.h"

namespace llm_inference {

QwenModel::QwenModel(const ModelConfig & config, const ModelWeights & weights)
    : config_(config),
      params_(parse_model_params(weights, config)),
      decoder_(config_, params_),
      lm_head_(params_.final_norm, params_.embed_tokens, config_.text.rms_norm_eps) {
}

const char * QwenModel::name() const {
    return "QwenModel";
}

Tensor QwenModel::prefill(const std::vector<int> & input_ids, RunState & state) const {
    return decoder_.prefill(input_ids, state);
}

Tensor QwenModel::forward(const Tensor & device_token_id, RunState & state) const {
    return decoder_.forward(device_token_id, state);
}

void QwenModel::forward_lm_head(const Tensor & device_hidden, const Tensor & device_token_out) const {
    lm_head_.forward(device_hidden, device_token_out);
}

} // namespace llm_inference
