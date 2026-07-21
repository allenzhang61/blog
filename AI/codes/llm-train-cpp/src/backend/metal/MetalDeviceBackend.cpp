#include "llm/backend/MetalBackend.hpp"

namespace llm {

DeviceType MetalBackend::type() const {
    return DeviceType::Metal;
}

std::string MetalBackend::name() const {
    // 与既有注册表命名约定保持一致（to_string(Metal) + "Backend"）。
    return "metalBackend";
}

} // namespace llm
