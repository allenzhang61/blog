//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeight.h"

CudaWeight::~CudaWeight() {
    reset();
}

CudaWeight::CudaWeight(CudaWeight &&other) noexcept
    : ptr(other.ptr), bytes(other.bytes), type(other.type) {
    other.ptr = nullptr;
    other.bytes = 0;
}

CudaWeight &CudaWeight::operator=(CudaWeight &&other) noexcept {
    if (this != &other) {
        reset();
        ptr = other.ptr;
        bytes = other.bytes;
        type = other.type;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    return *this;
}

void CudaWeight::reset() {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
    bytes = 0;
}
