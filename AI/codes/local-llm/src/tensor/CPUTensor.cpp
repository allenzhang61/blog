//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/CPUTensor.h"

#include "backend/cuda/common.h"

#include <utility>

#include <cuda_runtime.h>

CPUTensor CPUTensor::host_view(const void *host_ptr, std::vector<int64_t> shape, DType dt) {
    CPUTensor t;
    t.shape = std::move(shape);
    t.dtype = dt;
    t.nbytes = t.byte_size();
    t.data = const_cast<void *>(host_ptr);
    return t;
}

const int *CPUTensor::host_i32() const {
    return static_cast<const int *>(data);
}

GPUTensor CPUTensor::to_gpu(CudaScratch &scratch, const std::string &key,
                            const std::string &what) const {
    GPUTensor dst = GPUTensor::gpu_scratch(scratch, key, shape, dtype);
    to_gpu(dst.data, what);
    return dst;
}

void CPUTensor::to_gpu(void *device_ptr, const std::string &what) const {
    check_cuda(cudaMemcpy(device_ptr, data, byte_size(), cudaMemcpyHostToDevice), what);
}
