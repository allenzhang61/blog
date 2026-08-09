#include "CudaWeightCache.h"

#include "../cuda_common.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace llm_inference {

size_t CudaWeightCache::cache_limit_bytes() {
    const char * env = std::getenv("LLM_INFERENCE_CUDA_WEIGHT_CACHE_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 10.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

cudaDataType_t CudaWeightCache::cuda_type_for(const WeightData & weight) {
    if (weight.info->dtype == "BF16") {
        return CUDA_R_16BF;
    }
    if (weight.info->dtype == "F16") {
        return CUDA_R_16F;
    }
    if (weight.info->dtype == "F32") {
        return CUDA_R_32F;
    }
    throw std::runtime_error("暂不支持 CUDA dtype：" + weight.info->dtype + " tensor=" + weight.info->name);
}

size_t CudaWeightCache::dtype_size_for(const WeightData & weight) {
    if (weight.info->dtype == "BF16" || weight.info->dtype == "F16") {
        return sizeof(uint16_t);
    }
    if (weight.info->dtype == "F32") {
        return sizeof(float);
    }
    throw std::runtime_error("暂不支持 dtype：" + weight.info->dtype + " tensor=" + weight.info->name);
}

CudaWeightCache::CudaWeightCache() {
    check_cublas(cublasCreate(&handle), "cublasCreate 失败");
}

CudaWeightCache::~CudaWeightCache() {
    if (handle) {
        cublasDestroy(handle);
    }
}

CudaWeightCache & cuda_weight_cache() {
    static CudaWeightCache cache;
    return cache;
}

DeviceWeight * CudaWeightCache::cached_weight(const WeightData & weight) {
    auto found = items.find(weight.info->name);
    if (found != items.end()) {
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
    if (this->bytes + bytes > limit) {
        items.clear();
        this->bytes = 0;
    }

    DeviceWeight device;
    device.bytes = bytes;
    device.type = cuda_type_for(weight);
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc weight 失败 " + weight.info->name);
    check_cuda(cudaMemcpy(device.ptr, weight.data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy weight 失败 " + weight.info->name);
    auto [it, inserted] = items.emplace(weight.info->name, std::move(device));
    this->bytes += bytes;
    (void) inserted;
    return &it->second;
}

DeviceWeight * CudaWeightCache::cached_concat_weight(const std::string & name, const std::vector<WeightData> & weights) {
    auto found = items.find(name);
    if (found != items.end()) {
        return &found->second;
    }
    if (weights.empty()) {
        return nullptr;
    }
    const int64_t in_dim = weights[0].info->shape[1];
    int64_t total_rows = 0;
    for (const WeightData & weight : weights) {
        if (weight.info->dtype != "BF16" || weight.info->shape.size() != 2 || weight.info->shape[1] != in_dim) {
            return nullptr;
        }
        total_rows += weight.info->shape[0];
    }

    const size_t elems = static_cast<size_t>(total_rows) * static_cast<size_t>(in_dim);
    const size_t bytes = elems * sizeof(uint16_t);
    const size_t limit = cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (this->bytes + bytes > limit) {
        items.clear();
        this->bytes = 0;
    }

    std::vector<uint16_t> host;
    host.reserve(elems);
    for (const WeightData & weight : weights) {
        const size_t weight_elems = static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(in_dim);
        const auto * p = reinterpret_cast<const uint16_t *>(weight.data);
        host.insert(host.end(), p, p + weight_elems);
    }

    DeviceWeight device;
    device.bytes = bytes;
    device.type = CUDA_R_16BF;
    check_cuda(cudaMalloc(&device.ptr, bytes), "cudaMalloc concat weight 失败 " + name);
    check_cuda(cudaMemcpy(device.ptr, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy concat weight 失败 " + name);
    auto [it, inserted] = items.emplace(name, std::move(device));
    this->bytes += bytes;
    (void) inserted;
    return &it->second;
}

} // namespace llm_inference
