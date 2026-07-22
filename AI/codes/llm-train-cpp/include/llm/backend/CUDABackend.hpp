#pragma once

#include "llm/backend/Backend.hpp"

namespace llm {

// CUDA 后端的身份标识类。
// 仅负责在 BackendRegistry 中报告设备类型与名称；
// 实际的 CUDA 计算走 ops:: -> cuda:: 的算子分发路径。
class CUDABackend : public Backend {
public:
    // 返回 DeviceType::CUDA。
    DeviceType type() const override;

    // 返回后端名称，通常用于日志输出。
    std::string name() const override;
};

} // namespace llm
