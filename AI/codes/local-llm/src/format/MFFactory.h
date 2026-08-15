//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_MFFACTORY_H
#define LOCAL_LLM_MFFACTORY_H

#include <memory>
#include <string>

#include "format/MF.h"

// 根据路径打开具体模型文件格式：
//   - 目录：HFFile（config.json / tokenizer.json / safetensors）
//   - .gguf 文件：GgufFile
std::unique_ptr<MF> open_mf(const std::string &model_path);

#endif // LOCAL_LLM_MFFACTORY_H
