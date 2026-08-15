//
// Created by zhangyoulun on 9/8/2026.
//

#include "ModelFactory.h"

#include <stdexcept>
#include <utility>

#include "llm/model/qwen/QwenModel.h"
#include "llm/model/deepseek/DeepseekModel.h"

std::unique_ptr<BaseModel> create_model(const std::string &name,
                                        std::unique_ptr<MF> mf,
                                        int max_output_tokens,
                                        const SamplingConfig &sampling) {
    if (name == "qwen") {
        return std::make_unique<QwenModel>(std::move(mf), max_output_tokens, sampling);
    }
    if (name == "deepseek") {
        return std::make_unique<DeepseekModel>(std::move(mf), max_output_tokens, sampling);
    }
    throw std::runtime_error("未知模型名: " + name + "（当前支持: qwen, deepseek）");
}
