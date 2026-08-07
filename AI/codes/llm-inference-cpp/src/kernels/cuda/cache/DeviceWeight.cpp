#include "DeviceWeight.h"

namespace llm_inference {

namespace {

template <typename T>
void cuda_free_if_set(T * ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

} // namespace

DeviceWeight::~DeviceWeight() {
    cuda_free_if_set(ptr);
}

DeviceWeight::DeviceWeight(DeviceWeight && other) noexcept {
    ptr = other.ptr;
    bytes = other.bytes;
    type = other.type;
    other.ptr = nullptr;
    other.bytes = 0;
}

DeviceWeight & DeviceWeight::operator=(DeviceWeight && other) noexcept {
    if (this != &other) {
        cuda_free_if_set(ptr);
        ptr = other.ptr;
        bytes = other.bytes;
        type = other.type;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    return *this;
}

} // namespace llm_inference
