#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// 计算 shape 中所有维度的乘积，也就是张量元素总数。
int64_t product(const std::vector<int64_t>& shape);

// 将负数维度转换为正数维度，例如 -1 表示最后一维。
int64_t canonical_dim(int64_t dim, int64_t rank);

// 简单断言工具；条件不满足时抛出异常。
void check(bool cond, const std::string& message);

// 浮点近似比较工具，常用于测试数值是否足够接近。
void check_close(double a, double b, double tol, const std::string& message);

} // namespace llm
