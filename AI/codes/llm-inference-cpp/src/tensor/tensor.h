#pragma once

#include "../core/safetensors.h"
#include "../kernels/cpu/cpu_ops.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm_inference {

// 张量元素 dtype。权重可能是低精度存储（BF16/F16），激活统一用 F32。
enum class DType {
    F32,
    BF16,
    F16,
};

// 计算设备。推理支持 CPU 与 CUDA 两种后端。
enum class Device {
    CPU,
    CUDA,
};

// 从 safetensors 的 dtype 字符串解析出 DType。
DType dtype_from_string(const std::string & s);

// DType 转字符串（用于日志/错误信息）。
const char * dtype_to_string(DType dt);

// 统一张量抽象（无 autograd，推理专用）。
//
// 两种存储形态二选一：
//   1. owned：拥有一段 F32 数据（用于中间激活），可写。
//   2. external：引用外部只读内存（用于 mmap 权重，dtype 可为 BF16/F16/F32），只读。
//
// 该类刻意保持轻量：不持有 CUDA device 指针，设备端缓存仍由既有的
// CudaWeightCache（void*）机制管理，以保证重构不改变数值与性能行为。
class Tensor {
public:
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    Device device = Device::CPU;

    Tensor() = default;

    // 构造一个拥有数据的 F32 激活张量（全 0）。
    static Tensor zeros(std::vector<int64_t> shape);

    // 从已有 F32 数据构造 owned 张量（拷贝）。
    static Tensor from_vector(std::vector<int64_t> shape, std::vector<float> data);

    // 从 mmap 权重引用构造 external 只读张量（不拷贝）。
    static Tensor from_weight(const TensorRef & ref);

    // 元素总数。
    int64_t numel() const;

    // 是否拥有底层数据（owned）。
    bool is_owned() const { return owned_ != nullptr; }

    // 可写 F32 数据指针（仅 owned F32 有效，否则抛异常）。
    float * data();

    // 只读 F32 数据指针（仅 owned F32 有效，否则抛异常）。
    const float * data() const;

    // 拥有数据时返回底层 vector 引用（仅 owned F32 有效）。
    std::vector<float> & vec();
    const std::vector<float> & vec() const;

    // 原始只读字节起始地址（external/owned 均可用）。
    const void * raw() const;

    // 按 dtype 读取第 index 个元素并转为 float（支持 external 的 BF16/F16/F32）。
    float at(size_t index) const;

private:
    // owned 情况下持有的 F32 数据；为空表示 external。
    std::shared_ptr<std::vector<float>> owned_;
    // external 情况下引用的只读内存起始地址。
    const void * ext_ptr_ = nullptr;
};

} // namespace llm_inference
