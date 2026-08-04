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
