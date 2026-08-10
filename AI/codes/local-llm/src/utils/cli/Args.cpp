//
// Created by zhangyoulun on 8/8/2026.
//

#include "Args.h"

#include "utils/log/Log.h"

#include <stdexcept>

Args::Args(const int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        auto get_value = [&](const std::string &key_) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(key_ + " 缺少参数值");
            }
            return argv[++i];
        };

        if (key == "--model") {
            this->model = get_value(key);
        } else if (key == "--model-dir") {
            this->model_dir = get_value(key);
        } else if (key == "--max-output-tokens") {
            this->max_output_tokens = std::stoi(get_value(key));
        } else if (key == "--profile") {
            this->profile = true;
        } else if (key == "--profile-dir") {
            this->profile_dir = get_value(key);
        }
    }
}

void Args::DebugDump() {
    Log::debug("model: " + this->model);
    Log::debug("model_dir: " + this->model_dir);
    Log::debug("max_output_tokens: " + std::to_string(this->max_output_tokens));
    Log::debug("profile: " + std::string(this->profile ? "true" : "false"));
    Log::debug("profile_dir: " + this->profile_dir);
}
