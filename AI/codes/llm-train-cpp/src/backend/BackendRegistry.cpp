#include "llm/backend.hpp"

#include <cstdlib>

namespace llm {

Backend::~Backend() = default;

UnimplementedBackend::UnimplementedBackend(DeviceType type) : type_(type) {}
DeviceType UnimplementedBackend::type() const { return type_; }
std::string UnimplementedBackend::name() const { return to_string(type_) + "Backend"; }

Backend& BackendRegistry::get(Device device) {
    static CPUBackend cpu;
    static UnimplementedBackend cuda(DeviceType::CUDA);
    static UnimplementedBackend metal(DeviceType::Metal);
    if (device.type == DeviceType::CPU) return cpu;
    if (device.type == DeviceType::CUDA) {
        if (cuda_backend_available()) return cuda;
        throw std::runtime_error(cuda_backend_status());
    }
    if (device.type == DeviceType::Metal) {
        if (metal_backend_available()) return metal;
        throw std::runtime_error(metal_backend_status());
    }
    return cpu;
}

Device select_device(const std::string& backend) {
    if (backend.empty()) return {};
    return Device::parse(backend);
}

Device select_device_from_arg_or_env(const std::string& arg, const char* env_name) {
    if (!arg.empty()) return select_device(arg);
    const char* value = std::getenv(env_name);
    if (value == nullptr || std::string(value).empty()) return {};
    return select_device(value);
}

} // namespace llm
