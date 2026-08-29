//
// Created by zhangyoulun on 8/8/2026.
//

#include "common.h"

#include <stdexcept>

#include "utils/stats/CudaAllocTracker.h"

namespace {
// 进程级当前流。decode 时切到非阻塞流以支持 CUDA Graph capture；默认 0 号流。
cudaStream_t g_current_stream = nullptr;
} // namespace

void set_current_cuda_stream(cudaStream_t stream) { g_current_stream = stream; }
cudaStream_t get_current_cuda_stream() { return g_current_stream; }

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
    // 在当前流上做 async 拷贝再同步：既保证与 kernel 的流内顺序，又保持调用方看到的同步语义。
    cudaStream_t s = get_current_cuda_stream();
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s), what);
    check_cuda(cudaStreamSynchronize(s), what + "(sync)");
}

void cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, const std::string &what) {
    cudaStream_t s = get_current_cuda_stream();
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s), what);
    check_cuda(cudaStreamSynchronize(s), what + "(sync)");
}

void cuda_memcpy2d_d2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width_bytes, size_t height, const std::string &what) {
    check_cuda(cudaMemcpy2D(dst, dpitch, src, spitch, width_bytes, height,
                            cudaMemcpyDeviceToDevice),
               what);
}

void *cuda_malloc_device(size_t bytes, const std::string &what) {
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), what);
    record_cuda_alloc(ptr, bytes, classify_cuda_alloc(what), what);
    return ptr;
}

void cuda_free_device(void *ptr) {
    if (ptr) {
        record_cuda_free(ptr);
        cudaFree(ptr);
    }
}
