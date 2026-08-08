//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDA_COMMON_H
#define LOCAL_LLM_CUDA_COMMON_H

#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// 检查 CUDA Runtime API 返回值，失败时附带上下文信息抛异常。
void check_cuda(cudaError_t status, const std::string &what);

// 检查 cuBLAS API 返回值，失败时附带上下文信息抛异常。
void check_cublas(cublasStatus_t status, const std::string &what);

#endif // LOCAL_LLM_CUDA_COMMON_H
