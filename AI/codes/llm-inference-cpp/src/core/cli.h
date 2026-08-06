#pragma once

#include "common.h"

#include <string>
#include <vector>

namespace llm_inference {

// 命令行参数解析后的运行配置。
struct Args {
    // Hugging Face 模型 snapshot 目录。
    std::string model_dir;
    // 用户输入 prompt；为空时使用 DEFAULT_PROMPT_IDS。
    std::string prompt;
    // 逗号分隔 token ids 解析后的输入，用于绕过 tokenizer。
    std::vector<int> input_ids;
    // 最多生成的新 token 数。
    int max_new_tokens = 1;
    // 正式计时前的预热推理次数。
    int warmup_runs = 0;
    // 采样温度；当前主要记录参数，完整路径仍以 greedy 为主。
    float temperature = 0.7f;
    // 是否启用贪心解码路径。
    bool greedy = false;
    // 是否关闭 thinking；当前仅影响日志/语义记录。
    bool disable_thinking = false;
    // 是否输出 profile timing JSON。
    bool profile_timing = false;
    // 是否打印 safetensors 中的 tensor 列表。
    bool dump_tensors = false;
    // 运行设备；严格设备匹配，不回退。默认 CPU。
    Device device = Device::CPU;
};

// 解析 argc/argv，并在参数非法时抛出异常或打印 usage 退出。
Args parse_args(int argc, char ** argv);

} // namespace llm_inference
