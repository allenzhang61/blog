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
    this->data_ = const_cast<void *>(host_ptr);
}

GPUTensor CPUTensor::to_gpu(CudaScratch &scratch, const std::string &key,
                            const std::string &what) const {
    GPUTensor g_dst = GPUTensor(scratch, key, shape, dtype);
    check_cuda(cudaMemcpy(g_dst.data(), data_, byte_size(), cudaMemcpyHostToDevice), what);
    return g_dst;
}
