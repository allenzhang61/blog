//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_MODELFACTORY_H
#define LOCAL_LLM_MODELFACTORY_H

#include <memory>
#include <string>

#include "format/MF.h"
#include "llm/model/BaseModel.h"
#include "utils/sampling/Sampler.h"

// 按模型名构造对应的推理模型。
//   name          : "qwen"（后续将支持 "deepseek"）；未知 name 抛异常。
//   mf    : 已打开的模型文件/目录抽象（HF 或 GGUF）。
//   max_output_tokens : 每次请求的最大生成长度，用于 session KV cache 容量。
//   sampling      : 采样配置（temperature/top-k/top-p/repetition penalty）。
std::unique_ptr<BaseModel> create_model(const std::string &name,
                                        std::unique_ptr<MF> mf,
                                        int max_output_tokens,
                                        const SamplingConfig &sampling);

#endif //LOCAL_LLM_MODELFACTORY_H
