//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHT_H
#define LOCAL_LLM_CUDAWEIGHT_H

#include <cstddef>

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
    // 释放 device 端数据。
    ~CudaWeight();

    // 独占所有权，禁止拷贝。
    CudaWeight(const CudaWeight &) = delete;
    CudaWeight &operator=(const CudaWeight &) = delete;

    // 转移 device 指针所有权，避免重复 cudaFree。
    CudaWeight(CudaWeight &&other) noexcept;
    // 释放当前缓冲区并接管 other 的 device 指针。
    CudaWeight &operator=(CudaWeight &&other) noexcept;

    // 释放当前 device 内存并清零。
    void reset();
};

#endif // LOCAL_LLM_CUDAWEIGHT_H
