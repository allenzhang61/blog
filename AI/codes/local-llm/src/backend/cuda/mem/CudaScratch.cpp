//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaScratch.h"

#include "../common.h"

CudaScratch::~CudaScratch() {
    reset();
}

void *CudaScratch::ensure_bytes(const std::string &key, size_t required_bytes,
                                const std::string &name) {
    Buffer &buf = buffers_[key];
    if (buf.bytes >= required_bytes) {
        return buf.ptr;
    }
    if (buf.ptr) {
        cuda_free_device(buf.ptr);
        buf.ptr = nullptr;
        buf.bytes = 0;
    }
    buf.ptr = cuda_malloc_device(required_bytes, "cudaMalloc " + name + " 失败");
    buf.bytes = required_bytes;
    return buf.ptr;
}

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
