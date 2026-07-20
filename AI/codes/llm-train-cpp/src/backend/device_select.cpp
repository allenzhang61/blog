#include "llm/backend/device_select.hpp"

#include <cstdlib>

namespace llm {

Device select_device(const std::string& backend) {
    if (backend.empty()) {
        return {};
    }
    return Device::parse(backend);
}

Device select_device_from_arg_or_env(const std::string& arg, const char* env_name) {
    if (!arg.empty()) {
        return select_device(arg);
    }
    const char* value = std::getenv(env_name);
    if (value == nullptr || std::string(value).empty()) {
        return {};
    }
    return select_device(value);
}

} // namespace llm
