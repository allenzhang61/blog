#include "llm/backend/CPUBackend.hpp"

namespace llm {

DeviceType CPUBackend::type() const {
    return DeviceType::CPU;
}

std::string CPUBackend::name() const {
    return "CPUBackend";
}

} // namespace llm
