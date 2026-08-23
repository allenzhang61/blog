//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/CPUTensor.h"

#include "backend/cuda/common.h"
#include "tensor/GPUTensor.h"

#include <utility>

#include <cuda_runtime.h>

CPUTensor::CPUTensor(const void *host_ptr, std::vector<int64_t> shape, DType dt) {
    this->shape = std::move(shape);
    this->dtype = dt;
    this->nbytes = byte_size();
    this->data_ = host_ptr;
}

GPUTensor CPUTensor::to_gpu(CudaScratch &scratch, const std::string &key,
                            const std::string &what) const {
    GPUTensor g_dst = GPUTensor(scratch, key, shape, dtype);
    // 走当前流的 async+sync 拷贝，保证与 compute_stream 上的 kernel 正确定序（非阻塞流不与 0 号流互斥）。
    cuda_memcpy_h2d(g_dst.data(), data_, byte_size(), what);
    return g_dst;
}
