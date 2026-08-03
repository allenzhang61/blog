#include "llm_inference.h"

#include <chrono>
#include <iostream>

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

} // namespace llm_inference
