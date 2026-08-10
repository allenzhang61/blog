//
// Created by zhangyoulun on 9/8/2026.
//

#include "ModelFactory.h"

#include <stdexcept>

#include "llm/qwen/model/QwenModel.h"
#include "llm/deepseek/model/DeepseekModel.h"

std::unique_ptr<BaseModel> create_model(const std::string &name,
                                        const std::string &model_dir,
                                        int max_output_tokens,
                                        const SamplingConfig &sampling) {
    if (name == "qwen") {
        return std::make_unique<QwenModel>(model_dir, max_output_tokens, sampling);
    }
    if (name == "deepseek") {
        // deepseek 的 model_dir 直接是 .gguf 文件路径。
        return std::make_unique<DeepseekModel>(model_dir, max_output_tokens, sampling);
    }
    throw std::runtime_error("未知模型名: " + name + "（当前支持: qwen, deepseek）");
}
