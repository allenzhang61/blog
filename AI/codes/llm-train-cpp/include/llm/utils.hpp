#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// 计算 shape 中所有维度的乘积，也就是张量元素总数。
int64_t product(const std::vector<int64_t>& shape);

// 将负数维度转换为正数维度，例如 -1 表示最后一维。
int64_t canonical_dim(int64_t dim, int64_t rank);

} // namespace llm
