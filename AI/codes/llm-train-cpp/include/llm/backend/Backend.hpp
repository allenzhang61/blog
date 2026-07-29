#pragma once

#include "llm/device.hpp"

#include <string>

namespace llm {

namespace backend {

// 查询给定后端当前是否可用。
bool available(DeviceType type);

// 返回给定后端的状态说明。
std::string status(DeviceType type);

// 请求的后端不可用时抛出错误；CPU 恒可用。
void require_available(DeviceType type);

} // namespace backend

// 下面这 4 个函数是 CUDA / Metal 的底层 bridge，供 backend:: 实现复用。
// CPU 不需要单独 bridge，因为始终可用。
bool cuda_backend_available();
bool metal_backend_available();
std::string cuda_backend_status();
std::string metal_backend_status();

} // namespace llm
