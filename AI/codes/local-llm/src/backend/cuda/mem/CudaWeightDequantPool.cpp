//
// Created by zhangyoulun on 15/8/2026.
//

#include "CudaWeightDequantPool.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/Quant.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>

namespace {
CudaWeightDequantPool *g_dequant_pool = nullptr;
} // namespace

void set_global_cuda_weight_dequant_pool(CudaWeightDequantPool *pool) {
    g_dequant_pool = pool;
}

CudaWeightDequantPool *global_cuda_weight_dequant_pool() {
    return g_dequant_pool;
}

size_t CudaWeightDequantPool::cache_limit_bytes() {
    const char *env = std::getenv("LOCAL_LLM_CUDA_DEQUANT_POOL_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 1.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

size_t CudaWeightDequantPool::cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
}

void CudaWeightDequantPool::touch(std::unordered_map<std::string, Entry>::iterator it) {
    lru_.erase(it->second.lru_it);
    lru_.push_front(it->first);
    it->second.lru_it = lru_.begin();
}

void CudaWeightDequantPool::evict_until(size_t bytes) {
    const size_t limit = cache_limit_bytes();
    if (bytes_ + bytes > limit && !lru_.empty()) {
        check_cuda(cudaDeviceSynchronize(), "CudaWeightDequantPool LRU 淘汰前同步失败");
    }
    while (bytes_ + bytes > limit && !lru_.empty()) {
        const std::string victim = lru_.back();
        auto it = items_.find(victim);
        if (it != items_.end()) {
            bytes_ -= it->second.weight->bytes;
            items_.erase(it);
        }
        lru_.pop_back();
    }
}

CudaWeight CudaWeightDequantPool::cached_dequant(const CudaWeight &quant) {
    const size_t bytes = static_cast<size_t>(quant.num_elements) * sizeof(uint16_t);
    const size_t limit = cache_limit_bytes();
    if (bytes == 0 || bytes > limit) {
        throw std::runtime_error("dequant cache 单个权重超过阈值: " + quant.name);
    }

    const std::string key = quant.name + ".dequant.f16";
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = items_.find(key);
    if (found != items_.end()) {
        touch(found);
        return CudaWeight::make_view(found->second.weight->ptr, found->second.weight->bytes,
                                     found->second.weight->type, found->second.weight->dtype,
                                     found->second.weight->num_elements, key,
                                     found->second.weight);
    }

    evict_until(bytes);
    if (bytes_ + bytes > limit) {
        throw std::runtime_error("dequant cache LRU 淘汰后仍无法容纳: " + quant.name);
    }

    auto weight = std::make_shared<CudaWeight>(bytes, CUDA_R_16F, false, key);
    weight->dtype = DType::F16;
    weight->num_elements = quant.num_elements;
    weight->name = key;
    Quant::dequantize_to_f16(quant, static_cast<uint16_t *>(weight->ptr), quant.num_elements,
                             Quant::dtype_code(quant.dtype));

    lru_.push_front(key);
    items_.emplace(key, Entry{weight, lru_.begin()});
    bytes_ += bytes;
    return CudaWeight::make_view(weight->ptr, weight->bytes, weight->type,
                                 weight->dtype, weight->num_elements, key, weight);
}
