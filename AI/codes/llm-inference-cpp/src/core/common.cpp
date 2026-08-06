#include "common.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace llm_inference {

const char * MODEL_ID = "Qwen/Qwen3.5-4B";
const char * DEFAULT_PROMPT = "介绍一下 TCP 三次握手";

const std::vector<int> DEFAULT_PROMPT_IDS = {
    248045, 846, 198, 113552, 25804, 220, 110114, 119587,
    248046, 198, 248045, 74455, 198, 248068, 198,
};

double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void log(const std::string & message) {
    std::cerr << message << std::endl;
}

Device device_from_string(const std::string & name) {
    if (name == "cpu") {
        return Device::CPU;
    }
    if (name == "cuda" || name == "gpu") {
        return Device::CUDA;
    }
    throw std::runtime_error("未知设备：" + name + "（仅支持 cpu / cuda）");
}

const char * device_name(Device device) {
    return device == Device::CUDA ? "cuda" : "cpu";
}

} // namespace llm_inference
