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

    // 确保至少可存放 count 个 T，返回 device 数据指针。
    T * ensure(size_t count, const std::string & name);

    // 确保至少分配 required_bytes 字节，适合 byte/staging 类 buffer。
    T * ensure_bytes(size_t required_bytes, const std::string & name);

    // 释放当前 device 内存，并把容量清零。
    void reset();

    // 返回当前 device 数据指针；可能为空。
    T * data() const;

    // 允许在 CUDA/cuBLAS 调用中直接按裸指针使用。
    operator T *() const;

    // 返回当前已分配容量的字节数。
    size_t bytes() const;

private:
    T * ptr_ = nullptr;
    size_t bytes_ = 0;
};

extern template class CudaScratchBuffer<uint8_t>;
extern template class CudaScratchBuffer<uint16_t>;
extern template class CudaScratchBuffer<int>;
extern template class CudaScratchBuffer<float>;

} // namespace llm_inference
