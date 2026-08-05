#pragma once

#include "config.h"
#include "../safetensors/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_inference {
namespace ops {

// CPU/可选加速路径矩阵向量乘，输出 y。
void matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y);

} // namespace ops
} // namespace llm_inference
