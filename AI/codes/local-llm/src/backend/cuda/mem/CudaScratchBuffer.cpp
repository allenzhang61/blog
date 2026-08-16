//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaScratchBuffer.h"

#include "../common.h"

template <typename T>
CudaScratchBuffer<T>::~CudaScratchBuffer() {
    reset();
}

template <typename T>
CudaScratchBuffer<T>::CudaScratchBuffer(CudaScratchBuffer &&other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

template <typename T>
CudaScratchBuffer<T> &CudaScratchBuffer<T>::operator=(CudaScratchBuffer &&other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

template <typename T>
T *CudaScratchBuffer<T>::ensure(size_t count, const std::string &name) {
    const size_t required_bytes = count * sizeof(T);
    if (bytes_ >= required_bytes) {
        return ptr_;
    }
    reset();
    ptr_ = static_cast<T *>(cuda_malloc_device(required_bytes, "cudaMalloc " + name + " 失败"));
    bytes_ = required_bytes;
    return ptr_;
}

template <typename T>
void CudaScratchBuffer<T>::reset() {
    if (ptr_) {
        cuda_free_device(ptr_);
        ptr_ = nullptr;
    }
    bytes_ = 0;
}

template class CudaScratchBuffer<uint8_t>;
template class CudaScratchBuffer<uint16_t>;
template class CudaScratchBuffer<int>;
template class CudaScratchBuffer<float>;
