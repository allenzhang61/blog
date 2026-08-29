//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/StorageTensor.h"

#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "tensor/GPUTensor.h"

#include <memory>
#include <stdexcept>
#include <utility>

StorageTensor::StorageTensor(const uint8_t *disk_ptr, std::vector<int64_t> shape,
                       DType dt, size_t bytes) {
    this->shape = std::move(shape);
    this->dtype = dt;
    this->nbytes = bytes;
    this->data_ = disk_ptr;
}

StorageTensor StorageTensor::slice(size_t byte_offset, std::vector<int64_t> slice_shape,
                             size_t slice_bytes, std::string slice_name) const {
    StorageTensor s_view = *this;
    s_view.name = std::move(slice_name);
    s_view.shape = std::move(slice_shape);
    s_view.nbytes = slice_bytes;
    s_view.data_ = data_ + byte_offset;
    if (storage_name_.empty()) {
        s_view.storage_name_ = name;
        s_view.storage_shape_ = shape;
        s_view.storage_data_ = data_;
        s_view.storage_nbytes_ = nbytes;
        s_view.storage_byte_offset_ = byte_offset;
    } else {
        s_view.storage_name_ = storage_name_;
        s_view.storage_shape_ = storage_shape_;
        s_view.storage_data_ = storage_data_;
        s_view.storage_nbytes_ = storage_nbytes_;
        s_view.storage_byte_offset_ = storage_byte_offset_ + byte_offset;
    }
    return s_view;
}

GPUTensor StorageTensor::to_gpu(bool dequant) const {
    if (!dequant) {
        throw std::runtime_error("StorageTensor::to_gpu 暂不支持不反量化的 GPU 权重");
    }

    CudaWeightPool &pool = global_cuda_weight_pool();
    CudaWeight *cached = pool.cached_weight(*this);
    if (cached == nullptr) {
        throw std::runtime_error("StorageTensor::to_gpu 权重超过 CudaWeightPool 上限: " + name);
    }

    auto dequant_weight = std::make_shared<CudaWeight>(cached->try_dequant());
    GPUTensor g_view(dequant_weight, shape);
    g_view.pool_ = &pool;
    return g_view;
}
