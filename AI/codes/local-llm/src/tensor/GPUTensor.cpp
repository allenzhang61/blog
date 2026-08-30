//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/GPUTensor.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "tensor/CPUScratch.h"
#include "tensor/CPUTensor.h"

#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

void GPUTensor::init_from_weight(const CudaWeight &weight,
                                 std::vector<int64_t> shape) {
    this->shape = std::move(shape);
    this->dtype = weight.dtype;
    if (this->dtype == DType::F32 || this->dtype == DType::F16 ||
        this->dtype == DType::BF16 || this->dtype == DType::I32) {
        this->nbytes = byte_size();
    } else {
        // Quantized tensors are stored in block formats, so their physical byte
        // size is not numel * dense_dtype_size(dtype).
        this->nbytes = weight.bytes;
    }
    if (this->nbytes > weight.bytes) {
        throw std::runtime_error("GPUTensor CudaWeight shape 超出 weight 范围: " + weight.name);
    }
    this->name = weight.name;
    this->data_ = weight.ptr;
}

GPUTensor::GPUTensor(void *device_ptr, std::vector<int64_t> shape, DType dt, std::string name) {
    if (device_ptr == nullptr) {
        throw std::runtime_error("GPUTensor device view 需要非空 device data");
    }
    this->shape = std::move(shape);
    this->dtype = dt;
    this->nbytes = byte_size();
    this->name = std::move(name);
    this->data_ = device_ptr;
}

GPUTensor::GPUTensor(CudaWeight &&weight, std::vector<int64_t> shape) {
    this->owned_weight_ = std::make_shared<CudaWeight>(std::move(weight));
    init_from_weight(*owned_weight_, std::move(shape));
}

GPUTensor::GPUTensor(std::shared_ptr<CudaWeight> weight,
                     std::vector<int64_t> shape) {
    if (weight == nullptr) {
        throw std::runtime_error("GPUTensor CudaWeight shared view 需要非空 weight");
    }
    init_from_weight(*weight, std::move(shape));
    this->owned_weight_ = std::move(weight);
}

GPUTensor::GPUTensor(CudaScratch &scratch, const std::string &key,
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
            throw std::runtime_error(std::string("GPUTensor scratch 构造不支持 dtype: ") + dtype_name(dt));
    }

    this->shape = std::move(shape);
    this->dtype = dt;
    this->nbytes = byte_size();
    this->data_ = device_ptr;
}

GPUTensor::GPUTensor(const GPUTensor &parent, const size_t byte_offset,
                     std::vector<int64_t> shape) {
    if (parent.data_ == nullptr) {
        throw std::runtime_error("GPUTensor view 构造需要 parent device data: " + parent.name);
    }
    if (byte_offset > parent.nbytes) {
        throw std::runtime_error("GPUTensor view 构造 byte_offset 越界: " + parent.name);
    }
    this->shape = std::move(shape);
    this->dtype = parent.dtype;
    this->nbytes = byte_size();
    if (byte_offset + this->nbytes > parent.nbytes) {
        throw std::runtime_error("GPUTensor view 构造 shape 超出 parent 范围: " + parent.name);
    }
    this->data_ = static_cast<uint8_t *>(parent.data_) + byte_offset;
    this->pool_ = parent.pool_;
    this->owned_weight_ = parent.owned_weight_;
    this->weight_view_lease_ = parent.weight_view_lease_;
}

void GPUTensor::setdata_from_host(const void *host_data, size_t bytes, const std::string &what) const {
    if (data_ == nullptr) {
        throw std::runtime_error("GPUTensor::setdata 需要 device data: " + name);
    }
    cuda_memcpy_h2d(data_, host_data, bytes, what);
}

CPUTensor GPUTensor::to_host(CPUScratch &scratch, const std::string &key, const std::string &what) const {
    if (data_ == nullptr) {
        throw std::runtime_error("GPUTensor::to_host 需要 device data: " + name);
    }
    CPUTensor h_dst(scratch, key, shape, dtype);
    // 走当前流的 async+sync 拷贝：logits 在 compute_stream 上算出，必须在同流同步后再回读，避免读到未完成结果。
    cuda_memcpy_d2h(h_dst.data(), data_, byte_size(), what);
    return h_dst;
}
