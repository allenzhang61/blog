#pragma once

#include "llm/backend/Backend.hpp"

namespace llm {

// CPU 后端。
// 第一阶段的完整实现主要落在 CPU kernels 上。
class CPUBackend : public Backend {
public:
    // 返回 DeviceType::CPU。
    DeviceType type() const override;

    // 返回后端名称，通常用于日志输出。
    std::string name() const override;
};

} // namespace llm
