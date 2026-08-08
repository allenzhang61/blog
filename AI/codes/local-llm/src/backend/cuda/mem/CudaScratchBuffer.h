//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDASCRATCHBUFFER_H
#define LOCAL_LLM_CUDASCRATCHBUFFER_H

#include <cstddef>
#include <cstdint>
#include <string>

// 固定用途的 CUDA 临时缓冲区：只增不减，容量足够时复用已有 device 内存。
// 与具体模型无关，可用于任何前向过程中反复覆盖的中间结果 / staging 场景。
template <typename T>
class CudaScratchBuffer {
public:
    CudaScratchBuffer() = default;
    ~CudaScratchBuffer();

    // 独占所有权，禁止拷贝。
    CudaScratchBuffer(const CudaScratchBuffer &) = delete;
    CudaScratchBuffer &operator=(const CudaScratchBuffer &) = delete;

    CudaScratchBuffer(CudaScratchBuffer &&other) noexcept;
    CudaScratchBuffer &operator=(CudaScratchBuffer &&other) noexcept;

    // 确保至少能容纳 count 个元素，返回 device 指针；已够大时复用旧内存。
    T *ensure(size_t count, const std::string &name);

    // 确保至少分配 required_bytes 字节，适合 byte / staging 类 buffer。
    T *ensure_bytes(size_t required_bytes, const std::string &name);

    // 当前已分配的字节数。
    size_t bytes() const { return bytes_; }

    // 释放当前 device 内存，并把容量清零。
    void reset();

    // 允许在 CUDA / cuBLAS 调用中直接按裸指针使用。
    operator T *() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    size_t bytes_ = 0;
};

extern template class CudaScratchBuffer<uint8_t>;
extern template class CudaScratchBuffer<uint16_t>;
extern template class CudaScratchBuffer<int>;
extern template class CudaScratchBuffer<float>;

#endif // LOCAL_LLM_CUDASCRATCHBUFFER_H
