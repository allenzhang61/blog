//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeightPool.h"

#include "../common.h"
#include "backend/cuda/mem/Quant.h"
#include "utils/stats/CudaAllocTracker.h"
#include "utils/stats/WeightLoadTracker.h"

#include <chrono>
#include <memory>
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
        check_cuda(cudaMalloc(ptr, bytes), "cudaMalloc s_weight 失败 " + what);
        record_cuda_alloc(*ptr, bytes, CudaAllocKind::Weight, "s_weight " + what);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    check_cuda(cudaMalloc(ptr, bytes), "cudaMalloc s_weight 失败 " + what);
    const auto t1 = std::chrono::steady_clock::now();
    record_cuda_alloc(*ptr, bytes, CudaAllocKind::Weight, "s_weight " + what);
    out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// 计时版 H2D 拷贝：timed 为 true 时用 CUDA event 测 host->device 耗时（毫秒），
// 否则退化为普通同步拷贝、耗时返回 0，避免非 profile 路径产生额外开销。
void CudaWeightPool::memcpy_h2d_timed(void *dst, const void *src, size_t bytes,
                                      const std::string &what, bool timed, double &out_ms) {
    out_ms = 0.0;
    if (!timed) {
        cuda_memcpy_h2d(dst, src, bytes, "cudaMemcpy s_weight 失败 " + what);
        return;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream = get_current_cuda_stream();
    cudaEventRecord(start, stream);
    cuda_memcpy_h2d(dst, src, bytes, "cudaMemcpy s_weight 失败 " + what);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    out_ms = static_cast<double>(ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

cudaDataType_t CudaWeightPool::cuda_type_for(const StorageTensor &s_weight) {
    const DType dtype = s_weight.dtype;
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
                             " tensor=" + s_weight.name);
}

size_t CudaWeightPool::dtype_size_for(const StorageTensor &s_weight) {
    const DType dtype = s_weight.dtype;
    if (dtype == DType::BF16 || dtype == DType::F16) {
        return sizeof(uint16_t);
    }
    if (dtype == DType::F32) {
        return sizeof(float);
    }
    throw std::runtime_error(std::string("暂不支持 dtype：") + dtype_name(dtype) +
                             " tensor=" + s_weight.name);
}

CudaWeightPool::CudaWeightPool() {
    check_cublas(cublasCreate(&handle), "cublasCreate 失败");
}

CudaWeightPool::~CudaWeightPool() {
    if (handle) {
        cublasDestroy(handle);
    }
}

CudaWeight *CudaWeightPool::cached_weight(const StorageTensor &s_weight, bool use_storage_view) {
    auto found = items_.find(s_weight.name);
    if (found != items_.end()) {
        return found->second.get();
    }

    if (use_storage_view && s_weight.is_storage_slice()) {
        StorageTensor storage(s_weight.storage_data(), s_weight.storage_shape(),
                              s_weight.dtype, s_weight.storage_nbytes());
        storage.name = s_weight.storage_name();
        CudaWeight *base = cached_weight(storage);
        auto base_found = items_.find(storage.name);
        if (base == nullptr || base_found == items_.end()) {
            throw std::runtime_error("CudaWeightPool 创建 storage view 失败: " + s_weight.name);
        }
        auto view = CudaWeight::make_view(
            static_cast<uint8_t *>(base->ptr) + s_weight.storage_byte_offset(),
            s_weight.nbytes,
            cuda_type_for(s_weight),
            s_weight.dtype,
            Quant::num_elements(s_weight),
            s_weight.name,
            base_found->second);
        auto view_ptr = std::make_shared<CudaWeight>(std::move(view));
        auto [it, inserted] = items_.emplace(s_weight.name, std::move(view_ptr));
        (void) inserted;
        return it->second.get();
    }

    size_t bytes = s_weight.nbytes;
    if (!Quant::is_quantized_dtype(s_weight.dtype)) {
        size_t elems = 1;
        for (int64_t dim : s_weight.shape) {
            elems *= static_cast<size_t>(dim);
        }
        bytes = elems * dtype_size_for(s_weight);
    }
    auto device = std::make_shared<CudaWeight>();
    device->bytes = bytes;
    device->type = cuda_type_for(s_weight);
    device->dtype = s_weight.dtype;
    device->num_elements = Quant::num_elements(s_weight);
    device->name = s_weight.name;
    double malloc_ms = 0.0;
    cuda_malloc_timed(&device->ptr, bytes, s_weight.name, tracker_ != nullptr, malloc_ms);
    double h2d_ms = 0.0;
    memcpy_h2d_timed(device->ptr, s_weight.data(), bytes, s_weight.name, tracker_ != nullptr, h2d_ms);
    auto [it, inserted] = items_.emplace(s_weight.name, std::move(device));
    bytes_ += bytes;
    (void) inserted;
    if (tracker_) {
        // 分配与拷贝拆成两条事件，用 kind 区分（resident 均为本次入驻后的累计量）。
        tracker_->record(WeightLoadEventKind::Alloc, s_weight.name, bytes, malloc_ms, bytes_);
        tracker_->record(WeightLoadEventKind::Upload, s_weight.name, bytes, h2d_ms, bytes_);
    }
    return it->second.get();
}

CudaWeight *CudaWeightPool::find_cached_weight(const StorageTensor &s_weight) {
    auto found = items_.find(s_weight.name);
    if (found == items_.end()) {
        return nullptr;
    }
    return found->second.get();
}
