#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm {

// 设备类型：当前 CPU 是主要实现，CUDA / Metal 作为可选后端入口。
enum class DeviceType { CPU, CUDA, Metal };

// 张量数据类型：训练中主要使用 Float32，token id 使用 Int64。
enum class DType { Float32, Int64 };

// 将设备类型转成人类可读字符串。
std::string to_string(DeviceType type);

// 将数据类型转成人类可读字符串。
std::string to_string(DType dtype);

// 张量所在的逻辑设备。
// index 用来表示第几个设备，例如 cuda:0、metal:0。
struct Device {
    DeviceType type{DeviceType::CPU};
    int index{0};

    // 从字符串解析设备，例如 "cpu"、"cuda:0"、"metal"。
    static Device parse(const std::string& text);

    // 转成字符串形式，便于日志和错误信息展示。
    std::string str() const;
};

// 计算 shape 中所有维度的乘积，也就是张量元素总数。
int64_t product(const std::vector<int64_t>& shape);

// 将负数维度转换为正数维度，例如 -1 表示最后一维。
int64_t canonical_dim(int64_t dim, int64_t rank);

// 简单断言工具；条件不满足时抛出异常。
void check(bool cond, const std::string& message);

// 浮点近似比较工具，常用于测试数值是否足够接近。
void check_close(double a, double b, double tol, const std::string& message);

} // namespace llm
