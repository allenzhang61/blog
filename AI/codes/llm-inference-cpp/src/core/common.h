#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace llm_inference {

// 文件系统命名空间别名，统一用于模型目录和权重文件路径。
namespace fs = std::filesystem;

// 统一计时器类型，用于 load / prefill / decode 等阶段耗时统计。
using Clock = std::chrono::steady_clock;

// 运行设备。严格设备匹配模式：算子只在选定设备上执行，缺失实现即报错，不回退。
enum class Device { CPU, CUDA };

// 解析设备名（cpu/cuda），非法时抛异常。
Device device_from_string(const std::string & name);

// 返回设备名字符串（"cpu"/"cuda"）。
const char * device_name(Device device);

// 默认模型标识，主要用于日志输出。
extern const char * MODEL_ID;

// 未显式传入 prompt 时使用的中文默认输入。
extern const char * DEFAULT_PROMPT;

// 默认 prompt 对应的 token ids，用于绕过当前不完整的 tokenizer。
extern const std::vector<int> DEFAULT_PROMPT_IDS;

// 返回从 start 到当前时刻的秒级耗时。
double elapsed_s(Clock::time_point start);

// 输出一行运行日志到 stderr。
void log(const std::string & message);

} // namespace llm_inference
