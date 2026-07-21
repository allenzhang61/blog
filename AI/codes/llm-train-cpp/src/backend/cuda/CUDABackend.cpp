#include "llm/backend/Backend.hpp"

namespace llm {

bool cuda_backend_available() {
#if LLM_CPP_ENABLE_CUDA_COMPILED
    return true;
#else
    return false;
#endif
}

std::string cuda_backend_status() {
#if LLM_CPP_ENABLE_CUDA_COMPILED
    return "CUDA backend compiled; minimal CUDA runtime path is available";
#else
    return "CUDA backend is unavailable: project was not compiled with LLM_CPP_ENABLE_CUDA=ON or CUDA Toolkit was not found";
#endif
}

} // namespace llm
