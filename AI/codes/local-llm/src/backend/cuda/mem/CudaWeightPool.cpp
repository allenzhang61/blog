//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeightPool.h"

#include "../common.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

size_t CudaWeightPool::cache_limit_bytes() {
    const char *env = std::getenv("LOCAL_LLM_CUDA_WEIGHT_POOL_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 10.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

cudaDataType_t CudaWeightPool::cuda_type_for(const WeightData &weight) {
    const std::string &dtype = weight.info->dtype;
    if (dtype == "BF16") {
        return CUDA_R_16BF;
    }
    if (dtype == "F16") {
        return CUDA_R_16F;
    }
    if (dtype == "F32") {
        return CUDA_R_32F;
    }
    throw std::runtime_error("暂不支持 CUDA dtype：" + dtype + " tensor=" + weight.info->name);
}

size_t CudaWeightPool::dtype_size_for(const WeightData &weight) {
    const std::string &dtype = weight.info->dtype;
    if (dtype == "BF16" || dtype == "F16") {
        return sizeof(uint16_t);
    }
    if (dtype == "F32") {
        return sizeof(float);
    }
    throw std::runtime_error("暂不支持 dtype：" + dtype + " tensor=" + weight.info->name);
}

CudaWeightPool::CudaWeightPool() {
    check_cublas(cublasCreate(&handle), "cublasCreate 失败");
}

CudaWeightPool::~CudaWeightPool() {
    if (handle) {
        cublasDestroy(handle);
    }
}

CudaWeight *CudaWeightPool::cached_weight(const WeightData &weight) {
    auto found = items_.find(weight.info->name);
    if (found != items_.end()) {
        return &found->second;
    }

    size_t elems = 1;
    for (int64_t dim : weight.info->shape) {
        elems *= static_cast<size_t>(dim);
    }
    const size_t bytes = elems * dtype_size_for(weight);
    const size_t limit = cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (bytes_ + bytes > limit) {
        items_.clear();
        bytes_ = 0;
    }

    CudaWeight device;
    device.bytes = bytes;
    device.type = cuda_type_for(weight);
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc weight 失败 " + weight.info->name);
    check_cuda(cudaMemcpy(device.ptr, weight.data, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy weight 失败 " + weight.info->name);
    auto [it, inserted] = items_.emplace(weight.info->name, std::move(device));
    bytes_ += bytes;
    (void) inserted;
    return &it->second;
}

CudaWeight *CudaWeightPool::cached_concat_weight(const std::string &name,
                                                 const std::vector<WeightData> &weights) {
    auto found = items_.find(name);
    if (found != items_.end()) {
        return &found->second;
    }
    if (weights.empty()) {
        return nullptr;
    }

    // 要求：各权重均为二维、dtype 一致、列数一致，按行（shape[0]）拼接。
    const std::string &dtype = weights[0].info->dtype;
    if (weights[0].info->shape.size() != 2) {
        return nullptr;
    }
    const int64_t in_dim = weights[0].info->shape[1];
    const size_t elem_bytes = dtype_size_for(weights[0]);
    int64_t total_rows = 0;
    for (const WeightData &weight : weights) {
        if (weight.info->dtype != dtype || weight.info->shape.size() != 2 ||
            weight.info->shape[1] != in_dim) {
            return nullptr;
        }
        total_rows += weight.info->shape[0];
    }

    const size_t elems = static_cast<size_t>(total_rows) * static_cast<size_t>(in_dim);
    const size_t bytes = elems * elem_bytes;
    const size_t limit = cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (bytes_ + bytes > limit) {
        items_.clear();
        bytes_ = 0;
    }

    // 在 host 端把各权重按行顺序拷进连续 buffer，再整体上传。
    std::vector<uint8_t> host;
    host.reserve(bytes);
    for (const WeightData &weight : weights) {
        const size_t weight_bytes =
            static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(in_dim) * elem_bytes;
        host.insert(host.end(), weight.data, weight.data + weight_bytes);
    }

    CudaWeight device;
    device.bytes = bytes;
    device.type = cuda_type_for(weights[0]);
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc concat weight 失败 " + name);
    check_cuda(cudaMemcpy(device.ptr, host.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy concat weight 失败 " + name);
    auto [it, inserted] = items_.emplace(name, std::move(device));
    bytes_ += bytes;
    (void) inserted;
    return &it->second;
}
