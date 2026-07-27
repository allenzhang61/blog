#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm/tensor.hpp"

namespace llm::metal::detail {

bool runtime_available();
std::shared_ptr<TensorStorage> create_tensor_storage();
void fill_data_buffer(TensorStorage& storage, size_t count, float value);
void adamw_update(TensorStorage& param, TensorStorage& grad, TensorStorage& m, TensorStorage& v,
                  size_t count, float lr, float weight_decay, float beta1, float beta2,
                  float eps, float bias_correction1, float bias_correction2);

} // namespace llm::metal::detail
