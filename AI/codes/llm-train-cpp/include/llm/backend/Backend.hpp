#pragma once

#include "llm/core.hpp"

namespace llm {

// 计算后端的公共接口。
// CPU、CUDA、Metal 后端都通过这个抽象暴露自己的设备类型和名称。
class Backend {
public:
    // 使用虚析构函数，保证通过 Backend* 释放派生类时行为正确。
    virtual ~Backend();

    // 返回后端对应的设备类型，例如 CPU、CUDA 或 Metal。
    virtual DeviceType type() const = 0;

    // 返回便于日志展示的人类可读名称。
    virtual std::string name() const = 0;
};

// 查询 CUDA 后端当前是否可用。
bool cuda_backend_available();

// 查询 Metal 后端当前是否可用。
bool metal_backend_available();

// 返回 CUDA 后端状态说明，用于打印错误或诊断信息。
std::string cuda_backend_status();

// 返回 Metal 后端状态说明，用于打印错误或诊断信息。
std::string metal_backend_status();

} // namespace llm
