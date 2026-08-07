#include "QwenLmHead.h"

#include "../kernels/cuda/cuda_ops.h"

#include <stdexcept>

namespace llm_inference {

QwenLmHead::QwenLmHead(const WeightData & final_norm, const WeightData & embedding, float rms_norm_eps)
    : final_norm_(final_norm),
      embedding_(embedding),
      hidden_size_(static_cast<int>(embedding.info->shape[1])),
      rms_norm_eps_(rms_norm_eps) {
}

const char * QwenLmHead::name() const {
    return "QwenLmHead";
}

void QwenLmHead::forward(const Tensor & device_hidden, const Tensor & device_token_out) const {
    if (!cuda_final_norm_argmax_to_device(
            final_norm_,
            embedding_,
            device_hidden.data,
            hidden_size_,
            rms_norm_eps_,
            true,
            device_token_out.data)) {
        throw std::runtime_error("CUDA final norm + argmax 到设备失败。");
    }
}

} // namespace llm_inference
