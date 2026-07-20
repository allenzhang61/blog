#pragma once

#include "llm/core.hpp"

namespace llm {

// 根据字符串选择设备。
// 支持类似 "cpu"、"cuda"、"metal" 的输入。
Device select_device(const std::string& backend);

// 优先使用命令行参数选择设备；参数为空时再读取环境变量。
// 默认环境变量名是 LLM_CPP_BACKEND。
Device select_device_from_arg_or_env(const std::string& arg = "", const char* env_name = "LLM_CPP_BACKEND");

} // namespace llm
