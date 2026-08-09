#pragma once

#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_inference {

// 检查 CUDA Runtime API 返回值，失败时附带上下文信息抛异常。
void check_cuda(cudaError_t status, const std::string & what);

// 检查 cuBLAS API 返回值，失败时附带上下文信息抛异常。
void check_cublas(cublasStatus_t status, const std::string & what);

} // namespace llm_inference
