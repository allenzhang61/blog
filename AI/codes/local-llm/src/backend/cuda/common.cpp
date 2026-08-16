//
// Created by zhangyoulun on 8/8/2026.
//

#include "common.h"

#include <stdexcept>

void check_cuda(cudaError_t status, const std::string &what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(what + "：" + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const std::string &what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(what + "，cublasStatus=" + std::to_string(static_cast<int>(status)));
    }
}

void cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, const std::string &what) {
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), what);
}

void cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, const std::string &what) {
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), what);
}

void *cuda_malloc_device(size_t bytes, const std::string &what) {
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), what);
    return ptr;
}

void cuda_free_device(void *ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}
