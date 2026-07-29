#include "llm/backend/Backend.hpp"

#include <stdexcept>

namespace llm {

namespace backend {

bool available(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return true;
        case DeviceType::CUDA:
            return cuda_backend_available();
        case DeviceType::Metal:
            return metal_backend_available();
    }
    return false;
}

std::string status(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return "CPU backend is always available";
        case DeviceType::CUDA:
            return cuda_backend_status();
        case DeviceType::Metal:
            return metal_backend_status();
    }
    return "unknown backend";
}

void require_available(DeviceType type) {
    if (!available(type)) {
        throw std::runtime_error(status(type));
    }
}

} // namespace backend

} // namespace llm
