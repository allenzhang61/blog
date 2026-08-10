//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeight.h"

#include "../common.h"

CudaWeight::CudaWeight(size_t bytes, cudaDataType_t type, bool zero, const std::string &what)
    : bytes(bytes), type(type) {
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc " + what + " 失败");
    if (zero) {
        check_cuda(cudaMemset(ptr, 0, bytes), "cudaMemset " + what + " 失败");
    }
}

CudaWeight::~CudaWeight() {
    reset();
}

CudaWeight CudaWeight::make_view(void *ptr, size_t bytes, cudaDataType_t type) {
    CudaWeight w;
    w.ptr = ptr;
    w.bytes = bytes;
    w.type = type;
    w.owns_ = false; // 视图不拥有内存，析构不释放。
    return w;
}

CudaWeight::CudaWeight(CudaWeight &&other) noexcept
    : ptr(other.ptr), bytes(other.bytes), type(other.type), owns_(other.owns_) {
    other.ptr = nullptr;
    other.bytes = 0;
    other.owns_ = true;
}

CudaWeight &CudaWeight::operator=(CudaWeight &&other) noexcept {
    if (this != &other) {
        reset();
        ptr = other.ptr;
        bytes = other.bytes;
        type = other.type;
        owns_ = other.owns_;
        other.ptr = nullptr;
        other.bytes = 0;
        other.owns_ = true;
    }
    return *this;
}

void CudaWeight::reset() {
    if (ptr && owns_) {
        cudaFree(ptr);
    }
    ptr = nullptr;
    bytes = 0;
    owns_ = true;
}
