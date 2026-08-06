#include "cuda_scratch_buffer.h"

#include "cuda_common.h"

#include <cuda_runtime.h>

namespace llm_inference {

template <typename T>
CudaScratchBuffer<T>::~CudaScratchBuffer() {
    reset();
}

template <typename T>
CudaScratchBuffer<T>::CudaScratchBuffer(CudaScratchBuffer && other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

template <typename T>
CudaScratchBuffer<T> & CudaScratchBuffer<T>::operator=(CudaScratchBuffer && other) noexcept {
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
T * CudaScratchBuffer<T>::ensure(size_t count, const std::string & name) {
    return ensure_bytes(count * sizeof(T), name);
}

template <typename T>
T * CudaScratchBuffer<T>::ensure_bytes(size_t required_bytes, const std::string & name) {
    if (bytes_ >= required_bytes) {
        return ptr_;
    }
    reset();
    void * raw = nullptr;
    check_cuda(cudaMalloc(&raw, required_bytes), "cudaMalloc " + name + " 失败");
    ptr_ = static_cast<T *>(raw);
    bytes_ = required_bytes;
    return ptr_;
}

template <typename T>
void CudaScratchBuffer<T>::reset() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    bytes_ = 0;
}

template <typename T>
T * CudaScratchBuffer<T>::data() const {
    return ptr_;
}

template <typename T>
CudaScratchBuffer<T>::operator T *() const {
    return ptr_;
}

template <typename T>
size_t CudaScratchBuffer<T>::bytes() const {
    return bytes_;
}

template class CudaScratchBuffer<uint8_t>;
template class CudaScratchBuffer<uint16_t>;
template class CudaScratchBuffer<int>;
template class CudaScratchBuffer<float>;

} // namespace llm_inference
