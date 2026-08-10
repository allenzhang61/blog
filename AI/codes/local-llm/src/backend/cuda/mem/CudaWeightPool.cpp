//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeightPool.h"

#include "../common.h"
#include "../ops/kernel.cuh"
#include "utils/stats/WeightLoadTracker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

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
        check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy weight 失败 " + what);
        return;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy weight 失败 " + what);
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

cudaDataType_t CudaWeightPool::cuda_type_for(const WeightData &weight) {
    const DType dtype = weight.info->dtype;
    if (dtype == DType::BF16) {
        return CUDA_R_16BF;
    }
    if (dtype == DType::F16) {
        return CUDA_R_16F;
    }
    if (dtype == DType::F32) {
        return CUDA_R_32F;
    }
    throw std::runtime_error(std::string("暂不支持 CUDA dtype：") + dtype_name(dtype) +
                             " tensor=" + weight.info->name);
}

size_t CudaWeightPool::dtype_size_for(const WeightData &weight) {
    const DType dtype = weight.info->dtype;
    if (dtype == DType::BF16 || dtype == DType::F16) {
        return sizeof(uint16_t);
    }
    if (dtype == DType::F32) {
        return sizeof(float);
    }
    throw std::runtime_error(std::string("暂不支持 dtype：") + dtype_name(dtype) +
                             " tensor=" + weight.info->name);
}

CudaWeightPool::CudaWeightPool() {
    check_cublas(cublasCreate(&handle), "cublasCreate 失败");
}

CudaWeightPool::~CudaWeightPool() {
    if (handle) {
        cublasDestroy(handle);
    }
}

CudaWeight *CudaWeightPool::cached_weight(const WeightData &weight) {
    auto found = items_.find(weight.info->name);
    if (found != items_.end()) {
        return &found->second;
    }

    size_t elems = 1;
    for (int64_t dim : weight.info->shape) {
        elems *= static_cast<size_t>(dim);
    }
    const size_t bytes = elems * dtype_size_for(weight);
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
    double malloc_ms = 0.0;
    cuda_malloc_timed(&device.ptr, bytes, weight.info->name, tracker_ != nullptr, malloc_ms);
    double h2d_ms = 0.0;
    memcpy_h2d_timed(device.ptr, weight.data, bytes, weight.info->name, tracker_ != nullptr, h2d_ms);
    auto [it, inserted] = items_.emplace(weight.info->name, std::move(device));
    bytes_ += bytes;
    (void) inserted;
    if (tracker_) {
        // 分配与拷贝拆成两条事件，用 kind 区分（resident 均为本次入驻后的累计量）。
        tracker_->record(WeightLoadEventKind::Alloc, weight.info->name, bytes, malloc_ms, bytes_);
        tracker_->record(WeightLoadEventKind::Upload, weight.info->name, bytes, h2d_ms, bytes_);
    }
    return &it->second;
}

CudaWeight *CudaWeightPool::cached_concat_weight(const std::string &name,
                                                 const std::vector<WeightData> &weights) {
    auto found = items_.find(name);
    if (found != items_.end()) {
        return &found->second;
    }
    if (weights.empty()) {
        return nullptr;
    }

    // 要求：各权重均为二维、dtype 一致、列数一致，按行（shape[0]）拼接。
    const DType dtype = weights[0].info->dtype;
    if (weights[0].info->shape.size() != 2) {
        return nullptr;
    }
    const int64_t in_dim = weights[0].info->shape[1];
    const size_t elem_bytes = dtype_size_for(weights[0]);
    int64_t total_rows = 0;
    for (const WeightData &weight : weights) {
        if (weight.info->dtype != dtype || weight.info->shape.size() != 2 ||
            weight.info->shape[1] != in_dim) {
            return nullptr;
        }
        total_rows += weight.info->shape[0];
    }

    const size_t elems = static_cast<size_t>(total_rows) * static_cast<size_t>(in_dim);
    const size_t bytes = elems * elem_bytes;
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

    // 在 host 端把各权重按行顺序拷进连续 buffer，再整体上传。
    std::vector<uint8_t> host;
    host.reserve(bytes);
    for (const WeightData &weight : weights) {
        const size_t weight_bytes =
            static_cast<size_t>(weight.info->shape[0]) * static_cast<size_t>(in_dim) * elem_bytes;
        host.insert(host.end(), weight.data, weight.data + weight_bytes);
    }

    CudaWeight device;
    device.bytes = bytes;
    device.type = cuda_type_for(weights[0]);
    double malloc_ms = 0.0;
    cuda_malloc_timed(&device.ptr, bytes, name, tracker_ != nullptr, malloc_ms);
    double h2d_ms = 0.0;
    memcpy_h2d_timed(device.ptr, host.data(), bytes, name, tracker_ != nullptr, h2d_ms);
    auto [it, inserted] = items_.emplace(name, std::move(device));
    bytes_ += bytes;
    (void) inserted;
    if (tracker_) {
        tracker_->record(WeightLoadEventKind::Alloc, name, bytes, malloc_ms, bytes_);
        tracker_->record(WeightLoadEventKind::Upload, name, bytes, h2d_ms, bytes_);
    }
    return &it->second;
}

CudaWeight *CudaWeightPool::cached_q4k_weight(const std::string &name, const uint8_t *host_src,
                                              size_t src_bytes) {
    auto found = items_.find(name);
    if (found != items_.end()) {
        return &found->second;
    }
    const size_t limit = cache_limit_bytes();
    if (src_bytes > limit) {
        return nullptr;
    }
    if (bytes_ + src_bytes > limit) {
        if (tracker_) {
            tracker_->record(WeightLoadEventKind::EvictAll, "", bytes_, 0.0, 0);
        }
        items_.clear();
        bytes_ = 0;
    }

    CudaWeight device;
    device.bytes = src_bytes;
    device.type = CUDA_R_8I; // 标记：原始 Q4_K 字节，需反量化后才能 gemm。
    double malloc_ms = 0.0;
    cuda_malloc_timed(&device.ptr, src_bytes, name, tracker_ != nullptr, malloc_ms);
    double h2d_ms = 0.0;
    memcpy_h2d_timed(device.ptr, host_src, src_bytes, name, tracker_ != nullptr, h2d_ms);
    auto [it, inserted] = items_.emplace(name, std::move(device));
    bytes_ += src_bytes;
    (void) inserted;
    if (tracker_) {
        tracker_->record(WeightLoadEventKind::Alloc, name, src_bytes, malloc_ms, bytes_);
        tracker_->record(WeightLoadEventKind::Upload, name, src_bytes, h2d_ms, bytes_);
    }
    return &it->second;
}

CudaWeight CudaWeightPool::dequantize_q4k_to_f16(const CudaWeight &q4k, uint16_t *d_out_f16,
                                                 int64_t num_elements) {
    launch_dequantize_q4k_to_f16(static_cast<const uint8_t *>(q4k.ptr), d_out_f16, num_elements,
                                 nullptr);
    return CudaWeight::make_view(d_out_f16, static_cast<size_t>(num_elements) * sizeof(uint16_t),
                                 CUDA_R_16F);
}

CudaWeight *CudaWeightPool::cached_quant_weight(const std::string &name, const uint8_t *host_src,
                                                size_t src_bytes) {
    // 存储语义与 cached_q4k_weight 完全一致：常驻原始字节，type=CUDA_R_8I 标记。
    return cached_q4k_weight(name, host_src, src_bytes);
}

CudaWeight CudaWeightPool::dequantize_to_f16(const CudaWeight &quant, uint16_t *d_out_f16,
                                             int64_t num_elements, int ggml_type, void *stream) {
    const uint8_t *src = static_cast<const uint8_t *>(quant.ptr);
    switch (ggml_type) {
        case 0: // F32
            launch_f32_to_f16_copy(reinterpret_cast<const float *>(src), d_out_f16, num_elements, stream);
            break;
        case 6: // Q5_0
            launch_dequantize_q50_to_f16(src, d_out_f16, num_elements, stream);
            break;
        case 8: // Q8_0
            launch_dequantize_q80_to_f16(src, d_out_f16, num_elements, stream);
            break;
        case 12: // Q4_K
            launch_dequantize_q4k_to_f16(src, d_out_f16, num_elements, stream);
            break;
        case 14: // Q6_K
            launch_dequantize_q6k_to_f16(src, d_out_f16, num_elements, stream);
            break;
        default:
            throw std::runtime_error("dequantize_to_f16: 不支持的 GGML 类型码 " +
                                     std::to_string(ggml_type));
    }
    return CudaWeight::make_view(d_out_f16, static_cast<size_t>(num_elements) * sizeof(uint16_t),
                                 CUDA_R_16F);
}
