//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/GPUTensor.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"

#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

GPUTensor GPUTensor::gpu_scratch(CudaScratch &scratch, const std::string &key,
                                 std::vector<int64_t> shape, DType dt) {
    int64_t count = 0;
    if (!shape.empty()) {
        count = 1;
        for (int64_t d : shape) { count *= d; }
    }

    void *device_ptr = nullptr;
    switch (dt) {
        case DType::F32:
            device_ptr = scratch.ensure<float>(key, static_cast<size_t>(count));
            break;
        case DType::F16:
        case DType::BF16:
            device_ptr = scratch.ensure<uint16_t>(key, static_cast<size_t>(count));
            break;
        case DType::I32:
            device_ptr = scratch.ensure<int>(key, static_cast<size_t>(count));
            break;
        default:
            throw std::runtime_error(std::string("GPUTensor::gpu_scratch 不支持 dtype: ") + dtype_name(dt));
    }

    return GPUTensor::gpu_view(device_ptr, std::move(shape), dt);
}

GPUTensor GPUTensor::gpu_view(void *device_ptr, std::vector<int64_t> shape, DType dt) {
    GPUTensor t;
    t.shape = std::move(shape);
    t.dtype = dt;
    t.nbytes = t.byte_size();
    t.data = device_ptr;
    return t;
}

float *GPUTensor::gpu_f32() const {
    return static_cast<float *>(data);
}

int *GPUTensor::gpu_i32() const {
    return static_cast<int *>(data);
}

void GPUTensor::to_host(void *host_ptr, const std::string &what) const {
    if (data == nullptr) {
        throw std::runtime_error("GPUTensor::to_host 需要 device data: " + name);
    }
    check_cuda(cudaMemcpy(host_ptr, data, byte_size(), cudaMemcpyDeviceToHost), what);
}
