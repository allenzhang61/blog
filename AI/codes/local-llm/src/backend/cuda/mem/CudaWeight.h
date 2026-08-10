//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHT_H
#define LOCAL_LLM_CUDAWEIGHT_H

#include <cstddef>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

    CudaWeight() = default;
    // 在 device 上分配 bytes 字节内存；zero 为 true 时清零。
    // what 用于分配失败时的错误信息上下文。
    CudaWeight(size_t bytes, cudaDataType_t type, bool zero, const std::string &what);
    // 释放 device 端数据（仅当拥有所有权时）。
    ~CudaWeight();

    // 构造一个“非拥有”视图：包装一段已存在的 device 内存（如 scratch 反量化结果），
    // 析构时不释放。用于把临时 f16 buffer 传给 gemm_weight（后者只读 ptr/type/bytes）。
    static CudaWeight make_view(void *ptr, size_t bytes, cudaDataType_t type);

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
};

#endif // LOCAL_LLM_CUDAWEIGHT_H
