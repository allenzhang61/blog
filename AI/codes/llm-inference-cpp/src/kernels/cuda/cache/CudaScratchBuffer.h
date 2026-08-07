#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace llm_inference {

// 固定用途的 CUDA 临时缓冲区：按需扩容，容量足够时复用已有 device 内存。
template <typename T>
class CudaScratchBuffer {
public:
    CudaScratchBuffer() = default;
    ~CudaScratchBuffer();
    CudaScratchBuffer(const CudaScratchBuffer &) = delete;
    CudaScratchBuffer & operator=(const CudaScratchBuffer &) = delete;
    CudaScratchBuffer(CudaScratchBuffer && other) noexcept;
    CudaScratchBuffer & operator=(CudaScratchBuffer && other) noexcept;

    // 确保至少分配 required_bytes 字节，适合 byte/staging 类 buffer。
    T * ensure_bytes(size_t required_bytes, const std::string & name);

    // 释放当前 device 内存，并把容量清零。
    void reset();

    // 允许在 CUDA/cuBLAS 调用中直接按裸指针使用。
    operator T *() const;

private:
    T * ptr_ = nullptr;
    size_t bytes_ = 0;
};

extern template class CudaScratchBuffer<uint8_t>;
extern template class CudaScratchBuffer<uint16_t>;
extern template class CudaScratchBuffer<int>;
extern template class CudaScratchBuffer<float>;

} // namespace llm_inference
