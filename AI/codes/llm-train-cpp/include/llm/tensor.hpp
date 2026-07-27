#pragma once

#include "llm/device.hpp"
#include "llm/utils.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace llm {

struct TensorNode;

// CUDA / Metal Tensor 的内部 device storage。公共 Tensor API 通过 shared_ptr 持有它；
// 具体的分配、拷贝和释放逻辑由对应 runtime 注入。
struct TensorStorage {
    void* data{nullptr};   // 指向设备端数据缓冲区的指针。
    void* grad{nullptr};   // 指向设备端梯度缓冲区的指针。
    size_t data_count{0};  // 数据缓冲区的元素个数。
    size_t grad_count{0};  // 梯度缓冲区的元素个数。
    // 释放设备端缓冲区。
    void (*release)(TensorStorage& storage){nullptr};
    // 把 host 数据拷贝到设备端 data 缓冲区。
    void (*copy_data_from_host)(TensorStorage& storage, const std::vector<double>& host){nullptr};
    // 把设备端 data 缓冲区拷回 host。
    void (*copy_data_to_host)(TensorStorage& storage, std::vector<double>& host){nullptr};
    // 把 host 梯度拷贝到设备端 grad 缓冲区。
    void (*copy_grad_from_host)(TensorStorage& storage, const std::vector<double>& host){nullptr};
    // 把设备端 grad 缓冲区拷回 host。
    void (*copy_grad_to_host)(TensorStorage& storage, std::vector<double>& host){nullptr};
    // 用指定值填充设备端 grad 缓冲区。
    void (*fill_grad)(TensorStorage& storage, size_t count, float value){nullptr};

    TensorStorage() = default;                                     // 默认构造，缓冲区为空。
    TensorStorage(const TensorStorage&) = delete;            // 禁止拷贝构造，避免重复释放设备内存。
    TensorStorage& operator=(const TensorStorage&) = delete; // 禁止拷贝赋值，同上。
    ~TensorStorage();                                              // 析构时调用 release 回收设备内存。
};

// 轻量级张量对象。
// Tensor 只保存 shared_ptr<TensorNode>，实际数据、梯度和反向传播信息都在 TensorNode 中。
class Tensor {
public:
    // 指向真实张量存储和 autograd 节点的共享指针。
    std::shared_ptr<TensorNode> node;

    // 构造一个空 Tensor，主要用于占位或延迟赋值。
    Tensor();

    // 按 shape 创建张量，并自动分配数据缓冲区。
    Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32,
           Device device = {}, bool requires_grad = false);

    // 使用已有数据创建张量。
    Tensor(std::vector<int64_t> shape, std::vector<double> data,
           DType dtype = DType::Float32, Device device = {}, bool requires_grad = false);

    // 创建全 0 张量。
    static Tensor zeros(const std::vector<int64_t>& shape, Device device = {}, bool requires_grad = false);

    // 创建全 1 张量。
    static Tensor ones(const std::vector<int64_t>& shape, Device device = {}, bool requires_grad = false);

    // 创建正态分布随机张量，scale 控制初始化标准差。
    static Tensor randn(const std::vector<int64_t>& shape, double scale = 0.02,
                        Device device = {}, bool requires_grad = false);

    // 从 double 数组创建 Float32 语义的张量。
    static Tensor from_vector(const std::vector<double>& values, const std::vector<int64_t>& shape,
                              Device device = {}, bool requires_grad = false);

    // 从整数数组创建 Int64 语义的张量，常用于 token id 或 target id。
    static Tensor from_ints(const std::vector<int64_t>& values, const std::vector<int64_t>& shape,
                            Device device = {});

    // 返回张量形状，例如 [batch, seq_len, emb_dim]。
    const std::vector<int64_t>& shape() const;

    // 返回张量数据类型。
    DType dtype() const;

    // 返回张量所在设备。
    Device device() const;

    // 是否需要记录梯度。
    bool requires_grad() const;

    // 返回张量元素总数。
    int64_t numel() const;

    // 返回可修改的数据缓冲区。
    std::vector<double>& data();

    // 返回只读的数据缓冲区。
    const std::vector<double>& data() const;

    // 显式返回可修改的数据缓冲区，并标记 host data 需要同步到 CUDA device。
    std::vector<double>& mutable_data();

    // 返回梯度缓冲区；如果梯度尚未分配，实现层会补齐。
    std::vector<double>& grad() const;

    // 显式返回可修改的梯度缓冲区，并标记 host grad 需要同步到 CUDA device。
    std::vector<double>& mutable_grad() const;

    // 显式标记 host data 已被修改。
    void mark_data_host_dirty() const;

    // 显式标记 host grad 已被修改。
    void mark_grad_host_dirty() const;

    // 当张量只有一个元素时，取出这个标量值。
    double item() const;

    // 清空当前张量的梯度。
    void zero_grad();

    // 从当前张量开始执行反向传播。
    void backward();
};

// Tensor 的真实存储节点。
// parents 和 backward_fn 组成一个很小的动态计算图。
struct TensorNode {
    // 张量形状。
    std::vector<int64_t> shape;

    // 张量数据类型。
    // DType 只是记录"逻辑类型"的标签，底层存储统一是 double
    DType dtype{DType::Float32};

    // 张量所在设备。
    Device device{};

    // 前向计算得到的数据。
    std::vector<double> data;

    // 反向传播累积得到的梯度。
    std::vector<double> grad;

    // CUDA 专用 device storage。CPU/Metal 路径保持为空，具体定义在实现层。
    std::shared_ptr<TensorStorage> cuda_storage;

    // Metal 专用 device storage。CPU/CUDA 路径保持为空，具体定义在实现层。
    std::shared_ptr<TensorStorage> metal_storage;

    // host data 有新写入，下一次 GPU kernel 消费前需要同步到 device。
    bool host_data_dirty{false};

    // device data 有新写入，下一次 host 读取前需要同步到 host mirror。
    bool device_data_dirty{false};

    // host grad 有新写入，下一次 GPU kernel 消费前需要同步到 device。
    bool host_grad_dirty{false};

    // device grad 有新写入，下一次 host 读取前需要同步到 host mirror。
    bool device_grad_dirty{false};

    // 是否参与 autograd。
    bool requires_grad{false};

    // 当前张量依赖的父节点。
    std::vector<Tensor> parents;

    // 当前节点的局部反向传播函数。
    std::function<void()> backward_fn;
};

// 根据 shape 计算连续内存布局下的 strides。
std::vector<int64_t> strides_for(const std::vector<int64_t>& shape);

} // namespace llm
