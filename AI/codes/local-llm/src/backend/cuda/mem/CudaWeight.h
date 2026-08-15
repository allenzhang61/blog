//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHT_H
#define LOCAL_LLM_CUDAWEIGHT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "format/MF.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

class CudaWeightDequantPool;

// 单块已上传到 CUDA device 的内存缓冲区（RAII，独占所有权）。
// 与具体模型无关：只描述一段带 dtype 的 device 内存，析构时自动释放。
class CudaWeight {
public:
    // device 端数据指针。
    void *ptr = nullptr;
    // device 端缓冲区字节数。
    size_t bytes = 0;
    // cuBLAS 使用的数据类型。
    cudaDataType_t type = CUDA_R_32F;
    // 原始模型权重 dtype；量化权重反量化时需要用它分发 kernel。
    DType dtype = DType::F32;
    // 原始模型权重元素数；量化权重反量化时决定输出 f16 元素数。
    int64_t num_elements = 0;
    // 权重名称，用于派生 dequant LRU cache key。
    std::string name;
    CudaWeight() = default;
    // 在 device 上分配 bytes 字节内存；zero 为 true 时清零。
    // what 用于分配失败时的错误信息上下文。
    CudaWeight(size_t bytes, cudaDataType_t type, bool zero, const std::string &what);
    // 释放 device 端数据（仅当拥有所有权时）。
    ~CudaWeight();

    // 构造一个“非拥有”视图：包装一段已存在的 device 内存（如 scratch 反量化结果），
    // 析构时不释放。用于把临时 f16 buffer 传给 gemm_weight（后者只读 ptr/type/bytes）。
    static CudaWeight make_view(void *ptr, size_t bytes, cudaDataType_t type,
                                DType dtype = DType::F32, int64_t num_elements = 0,
                                std::string name = "",
                                std::shared_ptr<void> keep_alive = nullptr);

    // 普通权重返回自身 view；量化权重返回 dequant pool 中的 f16 view。
    CudaWeight try_dequant() const;

    // 独占所有权，禁止拷贝。
    CudaWeight(const CudaWeight &) = delete;
    CudaWeight &operator=(const CudaWeight &) = delete;

    // 转移 device 指针所有权，避免重复 cudaFree。
    CudaWeight(CudaWeight &&other) noexcept;
    // 释放当前缓冲区并接管 other 的 device 指针。
    CudaWeight &operator=(CudaWeight &&other) noexcept;

    // 释放当前 device 内存并清零。
    void reset();

private:
    // 是否拥有 ptr 的所有权：true 时析构/reset 会 cudaFree；视图（make_view）为 false。
    bool owns_ = true;
    // 非拥有 view 可通过它延长底层缓存条目的生命周期。
    std::shared_ptr<void> keep_alive_;
};

#endif // LOCAL_LLM_CUDAWEIGHT_H
