#include "llm/backend/Backend.hpp"
#include "../../kernels/metal/metal_ops.hpp"

namespace llm {

bool metal_backend_available() {
    return metal::available();
}

std::string metal_backend_status() {
    return metal::status();
}

} // namespace llm
