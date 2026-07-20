#include "llm/core.hpp"

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

int64_t product(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

int64_t canonical_dim(int64_t dim, int64_t rank) {
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        throw std::runtime_error("dim out of range");
    }
    return dim;
}

void check(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error("check failed: " + message);
    }
}

void check_close(double a, double b, double tol, const std::string& message) {
    if (std::fabs(a - b) > tol) {
        std::ostringstream oss;
        oss << message << " expected " << b << " got " << a;
        throw std::runtime_error(oss.str());
    }
}

} // namespace llm
