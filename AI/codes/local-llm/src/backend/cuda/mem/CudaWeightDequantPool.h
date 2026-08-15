//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHTDEQUANTPOOL_H
#define LOCAL_LLM_CUDAWEIGHTDEQUANTPOOL_H

#include "backend/cuda/mem/CudaWeight.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

// 反量化后的 f16 权重 LRU 缓存。与 CudaWeightPool 的原始权重缓存分离，
// 便于单独限制显存占用。
class CudaWeightDequantPool {
public:
    CudaWeightDequantPool() = default;

    CudaWeightDequantPool(const CudaWeightDequantPool &) = delete;
    CudaWeightDequantPool &operator=(const CudaWeightDequantPool &) = delete;

    CudaWeight cached_dequant(const CudaWeight &quant);

    size_t cached_bytes() const;

private:
    struct Entry {
        std::shared_ptr<CudaWeight> weight;
        std::list<std::string>::iterator lru_it;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> items_;
    std::list<std::string> lru_;
    size_t bytes_ = 0;

    static size_t cache_limit_bytes();
    void touch(std::unordered_map<std::string, Entry>::iterator it);
    void evict_until(size_t bytes);
};

void set_global_cuda_weight_dequant_pool(CudaWeightDequantPool *pool);
CudaWeightDequantPool *global_cuda_weight_dequant_pool();

#endif // LOCAL_LLM_CUDAWEIGHTDEQUANTPOOL_H
