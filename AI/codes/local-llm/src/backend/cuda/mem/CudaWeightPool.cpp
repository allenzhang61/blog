//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeightPool.h"

#include "../common.h"
#include "backend/cuda/mem/Quant.h"
#include "utils/stats/WeightLoadTracker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

CudaWeightPool *g_weight_pool = nullptr;

}

void set_global_cuda_weight_pool(CudaWeightPool *pool) {
    g_weight_pool = pool;
}

CudaWeightPool &global_cuda_weight_pool() {
    if (g_weight_pool == nullptr) {
        throw std::runtime_error("全局 CudaWeightPool 未初始化");
    }
    return *g_weight_pool;
}

// 计时版显存分配：cudaMalloc 是 host 侧同步调用，用 CPU 时钟（steady_clock）测更准确，
// 不走 CUDA event（后者测的是 stream 上的 GPU 时间线）。timed 为 false 时不计时、返回 0。
void CudaWeightPool::cuda_malloc_timed(void **ptr, size_t bytes, const std::string &what,
                                       bool timed, double &out_ms) {
    out_ms = 0.0;
    if (!timed) {
        check_cuda(cudaMalloc(ptr, bytes), "cudaMalloc weight 失败 " + what);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    check_cuda(cudaMalloc(ptr, bytes), "cudaMalloc weight 失败 " + what);
    const auto t1 = std::chrono::steady_clock::now();
    out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// 计时版 H2D 拷贝：timed 为 true 时用 CUDA event 测 host->device 耗时（毫秒），
// 否则退化为普通同步拷贝、耗时返回 0，避免非 profile 路径产生额外开销。
void CudaWeightPool::memcpy_h2d_timed(void *dst, const void *src, size_t bytes,
                                      const std::string &what, bool timed, double &out_ms) {
    out_ms = 0.0;
    if (!timed) {
        cuda_memcpy_h2d(dst, src, bytes, "cudaMemcpy weight 失败 " + what);
        return;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cuda_memcpy_h2d(dst, src, bytes, "cudaMemcpy weight 失败 " + what);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    out_ms = static_cast<double>(ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

size_t CudaWeightPool::cache_limit_bytes() {
    const char *env = std::getenv("LOCAL_LLM_CUDA_WEIGHT_POOL_GB");
    const double gb = env ? std::max(std::atof(env), 0.0) : 10.0;
    return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

cudaDataType_t CudaWeightPool::cuda_type_for(const MFTensorView &weight) {
    const DType dtype = weight.dtype;
    if (dtype == DType::BF16) {
        return CUDA_R_16BF;
    }
    if (dtype == DType::F16) {
        return CUDA_R_16F;
    }
    if (dtype == DType::F32) {
        return CUDA_R_32F;
    }
    if (Quant::is_quantized_dtype(dtype)) {
        return CUDA_R_8I;
    }
    throw std::runtime_error(std::string("暂不支持 CUDA dtype：") + dtype_name(dtype) +
                             " tensor=" + weight.name);
}

size_t CudaWeightPool::dtype_size_for(const MFTensorView &weight) {
    const DType dtype = weight.dtype;
    if (dtype == DType::BF16 || dtype == DType::F16) {
        return sizeof(uint16_t);
    }
    if (dtype == DType::F32) {
        return sizeof(float);
    }
    throw std::runtime_error(std::string("暂不支持 dtype：") + dtype_name(dtype) +
                             " tensor=" + weight.name);
}

CudaWeightPool::CudaWeightPool() {
    check_cublas(cublasCreate(&handle), "cublasCreate 失败");
}

CudaWeightPool::~CudaWeightPool() {
    if (handle) {
        cublasDestroy(handle);
    }
}

CudaWeight *CudaWeightPool::cached_weight(const MFTensorView &weight) {
    auto found = items_.find(weight.name);
    if (found != items_.end()) {
        return &found->second;
    }

    size_t bytes = weight.nbytes;
    if (!Quant::is_quantized_dtype(weight.dtype)) {
        size_t elems = 1;
        for (int64_t dim : weight.shape) {
            elems *= static_cast<size_t>(dim);
        }
        bytes = elems * dtype_size_for(weight);
    }
    const size_t limit = cache_limit_bytes();
    if (bytes > limit) {
        return nullptr;
    }
    if (bytes_ + bytes > limit) {
        if (tracker_) {
            tracker_->record(WeightLoadEventKind::EvictAll, "", bytes_, 0.0, 0);
        }
        items_.clear();
        bytes_ = 0;
    }

    CudaWeight device;
    device.bytes = bytes;
    device.type = cuda_type_for(weight);
    device.dtype = weight.dtype;
    device.num_elements = Quant::num_elements(weight);
    device.name = weight.name;
    double malloc_ms = 0.0;
    cuda_malloc_timed(&device.ptr, bytes, weight.name, tracker_ != nullptr, malloc_ms);
    double h2d_ms = 0.0;
    memcpy_h2d_timed(device.ptr, weight.data, bytes, weight.name, tracker_ != nullptr, h2d_ms);
    auto [it, inserted] = items_.emplace(weight.name, std::move(device));
    bytes_ += bytes;
    (void) inserted;
    if (tracker_) {
        // 分配与拷贝拆成两条事件，用 kind 区分（resident 均为本次入驻后的累计量）。
        tracker_->record(WeightLoadEventKind::Alloc, weight.name, bytes, malloc_ms, bytes_);
        tracker_->record(WeightLoadEventKind::Upload, weight.name, bytes, h2d_ms, bytes_);
    }
    return &it->second;
}
