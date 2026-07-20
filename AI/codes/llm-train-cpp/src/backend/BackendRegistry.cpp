#include "llm/backend/BackendRegistry.hpp"
#include "llm/backend/CPUBackend.hpp"
#include "llm/backend/UnimplementedBackend.hpp"

namespace llm {

Backend& BackendRegistry::get(Device device) {
    static CPUBackend cpu;
    static UnimplementedBackend cuda(DeviceType::CUDA);
    static UnimplementedBackend metal(DeviceType::Metal);
    if (device.type == DeviceType::CPU) {
        return cpu;
    }
    if (device.type == DeviceType::CUDA) {
        if (cuda_backend_available()) {
            return cuda;
        }
        throw std::runtime_error(cuda_backend_status());
    }
    if (device.type == DeviceType::Metal) {
        if (metal_backend_available()) {
            return metal;
        }
        throw std::runtime_error(metal_backend_status());
    }
    return cpu;
}

} // namespace llm
