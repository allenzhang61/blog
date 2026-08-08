//
// Created by zhangyoulun on 8/8/2026.
//

#include "Args.h"

#include "utils/log/Log.h"

Args::Args(const int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        auto get_value = [&](const std::string &key_) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(key_ + " 缺少参数值");
            }
            return argv[++i];
        };

        if (key == "--model-dir") {
            this->model_dir = get_value(key);
        }
    }
}

void Args::DebugDump() {
    Log::debug("model_dir: " + this->model_dir);
}
