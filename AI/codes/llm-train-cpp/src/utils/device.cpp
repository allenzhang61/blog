#include "llm/device.hpp"

#include <stdexcept>

namespace llm {

std::string to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return "cpu";
        case DeviceType::CUDA:
            return "cuda";
        case DeviceType::Metal:
            return "metal";
    }
    return "unknown";
}

std::string to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Int64:
            return "int64";
    }
    return "unknown";
}

Device Device::parse(const std::string& text) {
    if (text == "cpu") {
        return {DeviceType::CPU, 0};
    }
    if (text == "cuda" || text == "cuda:0") {
        return {DeviceType::CUDA, 0};
    }
    if (text == "metal" || text == "metal:0") {
        return {DeviceType::Metal, 0};
    }
    throw std::runtime_error("unknown device: " + text);
}

std::string Device::str() const {
    return to_string(type) + ":" + std::to_string(index);
}

} // namespace llm
