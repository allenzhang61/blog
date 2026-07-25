#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../tensor/tensor_cuda_storage.hpp"

namespace llm::metal::detail {

bool runtime_available();
std::shared_ptr<TensorCudaStorage> create_tensor_storage();
void fill_data_buffer(TensorCudaStorage& storage, size_t count, float value);
void adamw_update(TensorCudaStorage& param, TensorCudaStorage& grad, TensorCudaStorage& m, TensorCudaStorage& v,
                  size_t count, float lr, float weight_decay, float beta1, float beta2,
                  float eps, float bias_correction1, float bias_correction2);

} // namespace llm::metal::detail
