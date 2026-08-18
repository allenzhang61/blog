//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaScratch.h"

#include "../common.h"

CudaScratch::~CudaScratch() {
    reset();
}

template <typename T>
T *CudaScratch::ensure(const std::string &key, size_t count) {
    const size_t required_bytes = count * sizeof(T);
    Buffer &buf = buffers_[key];
    if (buf.bytes >= required_bytes) {
        return static_cast<T *>(buf.ptr);
    }
    if (buf.ptr) {
        cuda_free_device(buf.ptr);
        buf.ptr = nullptr;
        buf.bytes = 0;
    }
    buf.ptr = cuda_malloc_device(required_bytes, "cudaMalloc scratch[" + key + "] 失败");
    buf.bytes = required_bytes;
    return static_cast<T *>(buf.ptr);
}

// 前向过程中用到的元素类型：显式实例化，供其他 TU 链接。
template float *CudaScratch::ensure<float>(const std::string &, size_t);
template uint16_t *CudaScratch::ensure<uint16_t>(const std::string &, size_t);
template int *CudaScratch::ensure<int>(const std::string &, size_t);

size_t CudaScratch::total_bytes() const {
    size_t total = 0;
    for (const auto &kv : buffers_) {
        total += kv.second.bytes;
    }
    return total;
}

void CudaScratch::reset() {
    for (auto &kv : buffers_) {
        if (kv.second.ptr) {
            cuda_free_device(kv.second.ptr);
        }
    }
    buffers_.clear();
}
