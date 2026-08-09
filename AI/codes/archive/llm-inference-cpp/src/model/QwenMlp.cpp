#include "QwenMlp.h"

#include "../core/cuda_kernels.h"
#include "../kernels/cuda/cuda_common.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../kernels/cuda/cache/CudaWeightCache.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llm_inference {

QwenMlp::QwenMlp(const MlpWeights & weights, int layer_index)
    : weights_(weights), layer_index_(layer_index) {
}

const char * QwenMlp::name() const {
    return "QwenMlp";
}

Tensor QwenMlp::forward(const Tensor & device_x) const {
    const int hidden_dim = static_cast<int>(weights_.gate.info->shape[1]);
    auto & cache = cuda_weight_cache();
    cache.mlp_out_buffer.ensure_bytes(static_cast<size_t>(hidden_dim) * sizeof(float), "layer mlp out buffer");
    const int intermediate_dim = static_cast<int>(weights_.gate.info->shape[0]);
    if (weights_.gate.info->dtype != "BF16" || weights_.up.info->dtype != "BF16" || weights_.down.info->dtype != "BF16") {
        throw std::runtime_error("CUDA MLP 失败，layer=" + std::to_string(layer_index_));
    }
    DeviceWeight * gate_up_device = cache.cached_concat_weight(
        weights_.gate.info->name + "\n" + weights_.up.info->name,
        {weights_.gate, weights_.up});
    DeviceWeight * down_device = cache.cached_weight(weights_.down);
    if (!gate_up_device || !down_device) {
        throw std::runtime_error("CUDA MLP 失败，layer=" + std::to_string(layer_index_));
    }
    const size_t intermediate_float_bytes = static_cast<size_t>(intermediate_dim) * sizeof(float);
    cache.gate_up_buffer.ensure_bytes(static_cast<size_t>(intermediate_dim) * 2 * sizeof(float), "mlp gate up buffer");
    cache.prod_buffer.ensure_bytes(intermediate_float_bytes, "mlp prod buffer");
    cache.prod_lowp_buffer.ensure_bytes(static_cast<size_t>(intermediate_dim) * sizeof(uint16_t), "mlp prod bf16 buffer");

    WeightMeta combined_info = *weights_.gate.info;
    combined_info.name = weights_.gate.info->name + "+up";
    combined_info.shape[0] = static_cast<int64_t>(intermediate_dim) * 2;
    WeightData combined_ref {&combined_info, nullptr};
    cuda_weight_matvec_to_device(cache, combined_ref, *gate_up_device, device_x.data, gate_up_device->type, cache.gate_up_buffer);
    launch_silu_mul(cache.gate_up_buffer, cache.gate_up_buffer + intermediate_dim, cache.prod_buffer, intermediate_dim, nullptr);
    check_cuda(cudaGetLastError(), "launch_silu_mul 失败");
    cuda_float_to_lowp(cache.prod_buffer, cache.prod_lowp_buffer, intermediate_dim, down_device->type);
    check_cuda(cudaGetLastError(), "launch_float_to_bf16 失败");
    cuda_weight_matvec_to_device(cache, weights_.down, *down_device, cache.prod_lowp_buffer, down_device->type, cache.mlp_out_buffer);
    return {cache.mlp_out_buffer, hidden_dim, -1};
}

} // namespace llm_inference
