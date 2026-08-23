//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDA_COMMON_H
#define LOCAL_LLM_CUDA_COMMON_H

#include <cstddef>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// 检查 CUDA Runtime API 返回值，失败时附带上下文信息抛异常。
void check_cuda(cudaError_t status, const std::string &what);

// 检查 cuBLAS API 返回值，失败时附带上下文信息抛异常。
void check_cublas(cublasStatus_t status, const std::string &what);

// Host -> Device 拷贝，调用方不需要直接依赖 CUDA runtime 枚举。
void cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, const std::string &what);

// Device -> Host 拷贝，调用方不需要直接依赖 CUDA runtime 枚举。
void cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, const std::string &what);

// Device -> Device 的 2D 跨步拷贝（对应 cudaMemcpy2D + cudaMemcpyDeviceToDevice），
// dpitch/spitch 为目的/源每行字节跨距，width_bytes 为每行有效字节数，height 为行数。
void cuda_memcpy2d_d2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width_bytes, size_t height, const std::string &what);

// 分配 device 内存，失败时抛异常。
void *cuda_malloc_device(size_t bytes, const std::string &what);

// 释放 device 内存；用于析构 / reset 路径，不向外抛异常。
void cuda_free_device(void *ptr);

// 当前 CUDA 流：decode 阶段会切到一个非阻塞流，使 kernel 序列可被 stream capture（CUDA Graph）。
// 默认 nullptr（0 号默认流）。传给 launch 的 nullptr 会解析到此流；async memcpy 也用它。
void set_current_cuda_stream(cudaStream_t stream);
cudaStream_t get_current_cuda_stream();

#endif // LOCAL_LLM_CUDA_COMMON_H
