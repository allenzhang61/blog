#include "cli.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace llm_inference {

[[noreturn]] static void usage(const char * argv0, int code) {
    std::cerr
        << "用法:\n"
        << "  " << argv0 << " --model-dir DIR [--prompt TEXT] [options]\n\n"
        << "参数:\n"
        << "  --model-dir DIR          Hugging Face 模型 snapshot 目录，必填\n"
        << "  -p, --prompt TEXT        输入语句；当前仅内置默认 prompt tokenizer\n"
        << "  --input-ids IDS          逗号分隔 token ids，用于绕过 tokenizer\n"
        << "  --max-new-tokens N       最大生成 token 数；默认 1\n"
        << "  --temperature T          记录参数；当前完整路径只实现 greedy\n"
        << "  --greedy                 贪心解码\n"
        << "  --disable-thinking       当前仅影响日志；默认 prompt token ids 是 thinking=True 口径\n"
        << "  --warmup-runs N          正式统计前预热 N 次；默认 0\n"
        << "  --profile-timing         输出 PROFILE_TIMING_JSON\n"
        << "  --dump-tensors           打印 safetensors 中的 tensor 列表\n"
        << "  -h, --help               显示帮助\n";
    std::exit(code);
}

static int parse_int(const char * value, const std::string & name) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error(name + " 需要整数。");
    }
}

static float parse_float(const char * value, const std::string & name) {
    try {
        return std::stof(value);
    } catch (...) {
        throw std::runtime_error(name + " 需要数字。");
    }
}

static std::vector<int> parse_input_ids(const std::string & text) {
    std::vector<int> ids;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            ids.push_back(std::stoi(item));
        }
    }
    return ids;
}

Args parse_args(int argc, char ** argv) {
    Args args;
    args.prompt = DEFAULT_PROMPT;
    std::string positional_prompt;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string & name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " 缺少参数值。");
            }
            return argv[++i];
        };

        if (key == "-h" || key == "--help") {
            usage(argv[0], 0);
        } else if (key == "--model-dir") {
            args.model_dir = need_value(key);
        } else if (key == "-p" || key == "--prompt") {
            args.prompt = need_value(key);
        } else if (key == "--input-ids") {
            args.input_ids = parse_input_ids(need_value(key));
        } else if (key == "--max-new-tokens") {
            args.max_new_tokens = parse_int(need_value(key), key);
        } else if (key == "--temperature") {
            args.temperature = parse_float(need_value(key), key);
        } else if (key == "--greedy") {
            args.greedy = true;
        } else if (key == "--disable-thinking") {
            args.disable_thinking = true;
        } else if (key == "--warmup-runs") {
            args.warmup_runs = parse_int(need_value(key), key);
        } else if (key == "--profile-timing") {
            args.profile_timing = true;
        } else if (key == "--dump-tensors") {
            args.dump_tensors = true;
        } else if (key == "--device" || key == "--dtype" || key == "--cache-dir" || key == "--revision" ||
                   key == "--torch-profiler") {
            (void) need_value(key);
            log("提示：" + key + " 是 Python 版本参数，当前原生 C++ CPU 实现先忽略。");
        } else if (key == "--fast-decode" || key == "--static-cache" || key == "--nvtx") {
            log("提示：" + key + " 当前原生 C++ 实现先忽略。");
        } else if (!key.empty() && key[0] == '-') {
            throw std::runtime_error("未知参数：" + key);
        } else {
            if (!positional_prompt.empty()) {
                positional_prompt += " ";
            }
            positional_prompt += key;
        }
    }

    if (!positional_prompt.empty() && args.prompt == DEFAULT_PROMPT) {
        args.prompt = positional_prompt;
    }
    if (args.model_dir.empty()) {
        usage(argv[0], 1);
    }
    if (args.max_new_tokens <= 0) {
        throw std::runtime_error("--max-new-tokens 必须大于 0。");
    }
    if (args.warmup_runs < 0) {
        args.warmup_runs = 0;
    }
    if (!args.greedy) {
        log("提示：当前原生 C++ 完整推理路径只实现 greedy，已按 greedy 执行。");
        args.greedy = true;
    }
    return args;
}

} // namespace llm_inference
