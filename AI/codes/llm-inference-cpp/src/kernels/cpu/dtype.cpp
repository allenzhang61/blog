#include "cpu_ops.h"

#include <cstring>
#include <stdexcept>

namespace llm_inference {
namespace cpu {

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float f16_to_float(uint16_t h) {
    const uint16_t h_exp = h & 0x7C00u;
    const uint16_t h_sig = h & 0x03FFu;
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f = sign;
        } else {
            int exp = -1;
            uint16_t sig = h_sig;
            do {
                ++exp;
                sig <<= 1;
            } while ((sig & 0x0400u) == 0);
            sig &= 0x03FFu;
            f = sign | static_cast<uint32_t>(127 - 15 - exp) << 23 | static_cast<uint32_t>(sig) << 13;
        }
    } else if (h_exp == 0x7C00u) {
        f = sign | 0x7F800000u | (static_cast<uint32_t>(h_sig) << 13);
    } else {
        f = sign | (static_cast<uint32_t>((h_exp >> 10) + (127 - 15)) << 23) | (static_cast<uint32_t>(h_sig) << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

float tensor_value(const TensorRef & ref, size_t index) {
    if (ref.info->dtype == "BF16") {
        const auto * p = reinterpret_cast<const uint16_t *>(ref.data);
        return bf16_to_float(p[index]);
    }
    if (ref.info->dtype == "F16") {
        const auto * p = reinterpret_cast<const uint16_t *>(ref.data);
        return f16_to_float(p[index]);
    }
    if (ref.info->dtype == "F32") {
        const auto * p = reinterpret_cast<const float *>(ref.data);
        return p[index];
    }
    throw std::runtime_error("暂不支持 dtype：" + ref.info->dtype + " tensor=" + ref.info->name);
}

} // namespace cpu
} // namespace llm_inference
