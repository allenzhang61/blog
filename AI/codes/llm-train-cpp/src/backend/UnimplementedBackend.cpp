#include "llm/backend/UnimplementedBackend.hpp"

namespace llm {

UnimplementedBackend::UnimplementedBackend(DeviceType type) : type_(type) {
}

DeviceType UnimplementedBackend::type() const {
    return type_;
}

std::string UnimplementedBackend::name() const {
    return to_string(type_) + "Backend";
}

} // namespace llm
