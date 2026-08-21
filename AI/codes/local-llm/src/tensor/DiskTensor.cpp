//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/DiskTensor.h"

#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"

#include <stdexcept>
#include <utility>

DiskTensor DiskTensor::disk_view(const uint8_t *disk_ptr, std::vector<int64_t> shape,
                                 DType dt, size_t bytes) {
    DiskTensor t;
    t.shape = std::move(shape);
    t.dtype = dt;
    t.nbytes = bytes;
    t.data = disk_ptr;
    return t;
}

void DiskTensor::to_gpu() const {
    pool->cached_weight(*this);
}

GPUTensor DiskTensor::try_dequant() const {
    CudaWeight *w = pool->find_cached_weight(*this);
    if (w == nullptr) {
        throw std::runtime_error("DiskTensor::try_dequant 需要先调用 to_gpu(): " + name);
    }

    auto dequant = std::make_shared<CudaWeight>(w->try_dequant());
    GPUTensor view = GPUTensor::gpu_view(dequant->ptr, shape, dequant->dtype);
    view.name = dequant->name;
    view.nbytes = dequant->bytes;
    weight_view_lease = dequant;
    return view;
}

const void *DiskTensor::weight_gpu_data() const {
    GPUTensor weight = try_dequant();
    return weight.data;
}
