#include "QwenBlock.h"

#include "../kernels/cuda/cuda_ops.h"

#include <stdexcept>

namespace llm_inference {

QwenBlock::QwenBlock(const ModelConfig & config, const LayerWeights & weights, int layer_index)
    : config_(config), weights_(weights), layer_index_(layer_index) {
}

Tensor QwenBlock::allocate_output(const Tensor & device_x) const {
    const int hidden_size = config_.text.hidden_size;
    const int out_slot = device_x.slot == 0 ? 1 : 0;
    void * device_out = cuda_token_hidden_buffer(out_slot, hidden_size);
    if (!device_x.data || !device_out) {
        throw std::runtime_error("CUDA block hidden buffer 分配失败，layer=" + std::to_string(layer_index_));
    }
    return {device_out, hidden_size, out_slot};
}

} // namespace llm_inference
