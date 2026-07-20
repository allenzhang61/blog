#pragma once

#include "llm/backend/Backend.hpp"

namespace llm {

// 未实现后端的占位类。
// 用于保留 CUDA / Metal 等后端入口，让整体架构先稳定下来。
class UnimplementedBackend : public Backend {
public:
    // 记录这个占位对象代表哪一种设备类型。
    explicit UnimplementedBackend(DeviceType type);

    // 返回占位后端代表的设备类型。
    DeviceType type() const override;

    // 返回占位后端名称，便于错误信息说明。
    std::string name() const override;

private:
    // 当前占位后端对应的设备类型。
    DeviceType type_;
};

} // namespace llm
