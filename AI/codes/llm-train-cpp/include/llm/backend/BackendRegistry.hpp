#pragma once

#include "llm/backend/Backend.hpp"

namespace llm {

// 后端注册表。
// 根据 Device 选择实际使用的 Backend 实例。
class BackendRegistry {
public:
    // 获取指定设备对应的后端对象。
    // 如果请求的后端尚未实现或不可用，实现层会抛出错误。
    static Backend& get(Device device);
};

} // namespace llm
