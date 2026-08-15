//
// Created by zhangyoulun on 15/8/2026.
//

#include "format/MFFactory.h"

#include <filesystem>
#include <stdexcept>

#include "format/gguf/GgufFile.h"
#include "format/hf/HFFile.h"

std::unique_ptr<MF> open_mf(const std::string &model_path) {
    namespace fs = std::filesystem;
    const fs::path path(model_path);
    if (fs::is_directory(path)) {
        return std::make_unique<HFFile>(model_path);
    }
    if (fs::is_regular_file(path) && path.extension() == ".gguf") {
        return std::make_unique<GgufFile>(model_path);
    }
    throw std::runtime_error("无法识别模型路径格式: " + model_path + "（目录=HF，.gguf 文件=GGUF）");
}
