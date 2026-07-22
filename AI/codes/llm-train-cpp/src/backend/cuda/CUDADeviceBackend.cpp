#include "llm/backend/CUDABackend.hpp"

namespace llm {

DeviceType CUDABackend::type() const {
    return DeviceType::CUDA;
}

std::string CUDABackend::name() const {
    // 与既有注册表命名约定保持一致（to_string(CUDA) + "Backend"）。
    return "cudaBackend";
}

} // namespace llm
