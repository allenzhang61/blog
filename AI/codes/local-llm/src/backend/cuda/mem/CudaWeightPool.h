//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHTPOOL_H
#define LOCAL_LLM_CUDAWEIGHTPOOL_H

#include "CudaWeight.h"

#include "llm/qwen/QwenWeights.h" // 仅依赖其中通用的 WeightData / WeightMeta

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>

// 通用的 CUDA 权重缓存：持有 cuBLAS handle，并按 tensor 名称惰性地把 host 端
// （mmap）权重上传到 device 后缓存复用。与具体模型结构无关——只依赖通用的
// WeightData（safetensors tensor 引用）。超过字节上限时整体清空重来。
//
// 注意：本类只负责“权重”这一类持久 device 内存；前向过程中反复覆盖的临时中间
// 结果请用 CudaScratchBuffer，二者职责分离。
class CudaWeightPool {
public:
    // 创建 cuBLAS handle。
    CudaWeightPool();
    // 释放权重缓存和 cuBLAS handle。
    ~CudaWeightPool();

    CudaWeightPool(const CudaWeightPool &) = delete;
    CudaWeightPool &operator=(const CudaWeightPool &) = delete;

    // 全局复用的 cuBLAS handle。
    cublasHandle_t handle = nullptr;

    // 获取普通 device 权重缓存；首次访问时从 mmap host 权重上传到 GPU。
    // 单个权重超过上限时返回 nullptr。
    CudaWeight *cached_weight(const WeightData &weight);

    // 将多个二维权重按行拼接后上传并缓存，用于合并 projection（如 QKV / gate+up）。
    // 要求各权重 dtype 相同、列数（shape[1]）一致；不满足或超限时返回 nullptr。
    CudaWeight *cached_concat_weight(const std::string &name, const std::vector<WeightData> &weights);

    // 已缓存权重的总字节数。
    size_t cached_bytes() const { return bytes_; }

private:
    // 按 tensor 名称或组合名称索引的 device 权重缓存。
    std::unordered_map<std::string, CudaWeight> items_;
    // items_ 中已缓存权重的总字节数。
    size_t bytes_ = 0;

    // 返回本进程允许缓存的 CUDA 权重总字节数（可由环境变量覆盖）。
    static size_t cache_limit_bytes();

    // 将 safetensors dtype 映射为 CUDA / cuBLAS dtype。
    static cudaDataType_t cuda_type_for(const WeightData &weight);

    // 返回当前支持 dtype 的单元素字节数。
    static size_t dtype_size_for(const WeightData &weight);
};

#endif // LOCAL_LLM_CUDAWEIGHTPOOL_H
