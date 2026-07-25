#pragma once

#include <string>

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

} // namespace llm
