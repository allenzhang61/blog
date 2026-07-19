#include "llm/backend.hpp"

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
    if (device.type == DeviceType::CUDA) throw std::runtime_error("CUDA backend is not implemented yet; use cpu");
    if (device.type == DeviceType::Metal) throw std::runtime_error("Metal backend is not implemented yet; use cpu");
    return cpu;
}

} // namespace llm
