#pragma once

#include "llm/backend/Backend.hpp"

namespace llm {

// Metal 后端的身份标识类。
// 仅负责在 BackendRegistry 中报告设备类型与名称；
// 实际的 Metal 计算走 ops:: -> metal:: 的算子分发路径。
class MetalBackend : public Backend {
public:
    // 返回 DeviceType::Metal。
    DeviceType type() const override;

    // 返回后端名称，通常用于日志输出。
    std::string name() const override;
};

} // namespace llm
