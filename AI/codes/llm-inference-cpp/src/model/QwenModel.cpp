#include "QwenModel.h"

#include "../kernels/cuda/cuda_ops.h"

#include <algorithm>
#include <stdexcept>

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

std::vector<int> QwenModel::generate(
        RunState & state,
        const Args & args,
        const std::vector<int> & input_ids,
        Timing & timing) const {
    std::vector<int> generated;
    const int max_new_tokens = args.max_new_tokens;
    void * generated_device = cuda_token_id_buffer(max_new_tokens);
    if (!generated_device) {
        throw std::runtime_error("CUDA 生成 token 缓冲区分配失败。");
    }

    const auto prefill_start = Clock::now();
    Tensor hidden = decoder_.prefill(input_ids, state);
    if (!cuda_synchronize_device()) {
        throw std::runtime_error("CUDA 同步失败。");
    }
    timing.prefill_s = elapsed_s(prefill_start);

    const auto decode_start = Clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
        Tensor token_slot {static_cast<int *>(generated_device) + i, 1, -1};
        lm_head_.forward(hidden, token_slot);
        if (i + 1 < max_new_tokens) {
            hidden = forward(token_slot, state);
        }
    }
    if (!cuda_copy_generated_tokens_to_host(generated_device, max_new_tokens, generated)) {
        throw std::runtime_error("CUDA 生成 token 拷回主机失败。");
    }
    timing.decode_total_s += elapsed_s(decode_start);
    if (config_.text.eos_token_id >= 0) {
        const auto eos = std::find(generated.begin(), generated.end(), config_.text.eos_token_id);
        if (eos != generated.end()) {
            generated.resize(static_cast<size_t>(std::distance(generated.begin(), eos)) + 1);
        }
    }
    timing.generated_tokens += static_cast<int>(generated.size());
    timing.generated_ids.insert(timing.generated_ids.end(), generated.begin(), generated.end());
    return generated;
}

Tensor QwenModel::forward(const Tensor & device_token_id, RunState & state) const {
    return decoder_.forward(device_token_id, state);
}

} // namespace llm_inference
