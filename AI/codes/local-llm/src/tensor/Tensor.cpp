//
// Created by zhangyoulun on 15/8/2026.
//

#include "tensor/Tensor.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"

#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

const char *dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q5_0: return "Q5_0";
        case DType::Q5_1: return "Q5_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q3_K: return "Q3_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::Q8_K: return "Q8_K";
        case DType::I32: return "I32";
        case DType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

bool is_supported_dtype(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::Q4_K:
        case DType::Q5_0:
        case DType::Q6_K:
        case DType::Q8_0:
            return true;
        // 以下量化类型能被识别，但尚未实现反量化 kernel，暂不启用。
        // 要启用需先在 Quant::dequantize_to_f16 补对应 launch_dequantize_*_to_f16。
        // case DType::Q5_K: // 唯一实战常见（Q5_K_M），后续最该优先补
        // case DType::Q4_0: // legacy，已被 K-quant 取代，新模型基本不发布
        // case DType::Q4_1: // legacy，几乎绝迹
        // case DType::Q5_1: // legacy，少见
        // case DType::Q8_1: // legacy，少见
        // case DType::Q2_K: // 极限压缩档，质量损失大，小众
        // case DType::Q3_K: // 极限压缩档，小众
        // case DType::Q8_K: // 多为 llama.cpp 内部中间格式，权重文件基本不落盘
        default:
            return false;
    }
}

namespace {

size_t dtype_byte_size(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::F16:
        case DType::BF16:
            return 2;
        default:
            throw std::runtime_error(std::string("Tensor copy 不支持 dtype: ") + dtype_name(dt));
    }
}

void check_tensor_cuda(cudaError_t status, const std::string &what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(what + ": " + cudaGetErrorString(status));
    }
}

const void *host_or_disk_data(const Tensor &t) {
    if (t.has_location(TensorLocation::CpuMem)) {
        return t.cpu_data;
    }
    if (t.has_location(TensorLocation::DiskMmap)) {
        return t.disk_data;
    }
    return nullptr;
}

} // namespace

TensorLocation operator|(TensorLocation a, TensorLocation b) {
    return static_cast<TensorLocation>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

void Tensor::mark_location(TensorLocation loc) const {
    locations |= static_cast<uint32_t>(loc);
}

bool Tensor::has_location(TensorLocation loc) const {
    return (locations & static_cast<uint32_t>(loc)) != 0;
}

Tensor Tensor::gpu_scratch(CudaScratch &scratch, const std::string &key,
                      std::vector<int64_t> shape, DType dt) {
    int64_t count = 0;
    if (!shape.empty()) {
        count = 1;
        for (int64_t d : shape) { count *= d; }
    }

    void *device_ptr = nullptr;
    switch (dt) {
        case DType::F32:
            device_ptr = scratch.ensure<float>(key, static_cast<size_t>(count));
            break;
        case DType::F16:
        case DType::BF16:
            device_ptr = scratch.ensure<uint16_t>(key, static_cast<size_t>(count));
            break;
        case DType::I32:
            device_ptr = scratch.ensure<int>(key, static_cast<size_t>(count));
            break;
        default:
            throw std::runtime_error(std::string("Tensor::gpu_scratch 不支持 dtype: ") + dtype_name(dt));
    }

    return gpu_view(device_ptr, std::move(shape), dt);
}

Tensor Tensor::gpu_view(void *device_ptr, std::vector<int64_t> shape, DType dt) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = dt;
    t.gpu_data = device_ptr;
    t.mark_location(TensorLocation::GpuMem);
    return t;
}

Tensor Tensor::host_view(const void *host_ptr, std::vector<int64_t> shape, DType dt) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = dt;
    t.cpu_data = const_cast<void *>(host_ptr);
    t.mark_location(TensorLocation::CpuMem);
    return t;
}

float *Tensor::gpu_f32() const {
    return static_cast<float *>(gpu_data);
}

int *Tensor::gpu_i32() const {
    return static_cast<int *>(gpu_data);
}

const int *Tensor::host_i32() const {
    return static_cast<const int *>(cpu_data);
}

void Tensor::to_gpu(CudaScratch &scratch, const std::string &key, const std::string &what) {
    const void *src = host_or_disk_data(*this);
    if (!src) {
        throw std::runtime_error("Tensor::to_gpu 需要 CpuMem 或 DiskMmap 源: " + name);
    }

    Tensor dst = gpu_scratch(scratch, key, shape, dtype);
    check_tensor_cuda(cudaMemcpy(dst.gpu_data, src, byte_size(), cudaMemcpyHostToDevice), what);
    gpu_data = dst.gpu_data;
    mark_location(TensorLocation::GpuMem);
}

void Tensor::to_gpu(void *device_ptr, const std::string &what) const {
    const void *src = host_or_disk_data(*this);
    if (!src) {
        throw std::runtime_error("Tensor::to_gpu 需要 CpuMem 或 DiskMmap 源: " + name);
    }
    check_tensor_cuda(cudaMemcpy(device_ptr, src, byte_size(), cudaMemcpyHostToDevice), what);
}

void Tensor::to_host(void *host_ptr, const std::string &what) const {
    if (!has_location(TensorLocation::GpuMem) || !gpu_data) {
        throw std::runtime_error("Tensor::to_host 需要 GpuMem 源: " + name);
    }
    check_tensor_cuda(cudaMemcpy(host_ptr, gpu_data, byte_size(), cudaMemcpyDeviceToHost), what);
}

size_t Tensor::byte_size() const {
    return static_cast<size_t>(numel()) * dtype_byte_size(dtype);
}

Tensor Tensor::try_dequant() const {
    CudaWeight *w = pool->find_cached_weight(*this);
    if (w == nullptr) {
        throw std::runtime_error("Tensor::try_dequant 需要先调用 to_gpu(): " + name);
    }

    auto dequant = std::make_shared<CudaWeight>(w->try_dequant());
    Tensor view = *this;
    view.name = dequant->name;
    dtype_dequant = dequant->dtype;
    nbytes_dequant = dequant->bytes;
    gpu_data_dequant = dequant->ptr;
    weight_view_lease = dequant;
    view.dtype_dequant = dtype_dequant;
    view.nbytes_dequant = nbytes_dequant;
    view.gpu_data_dequant = gpu_data_dequant;
    view.weight_view_lease = std::move(dequant);
    view.mark_location(TensorLocation::GpuMem);
    return view;
}

const void *Tensor::weight_gpu_data() const {
    Tensor weight = try_dequant();
    weight_view_lease = weight.weight_view_lease;
    return weight.gpu_data_dequant;
}

int64_t Tensor::numel() const {
    if (shape.empty()) { return 0; }
    int64_t n = 1;
    for (int64_t d : shape) { n *= d; }
    return n;
}

int64_t Tensor::rows() const {
    if (shape.empty()) { return 0; }
    int64_t r = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) { r *= shape[i]; }
    return r;
}

int64_t Tensor::cols() const {
    return shape.empty() ? 0 : shape.back();
}

// 注意：Tensor::to_gpu() 的权重上传实现定义在
// backend/cuda/mem/CudaWeightPool.cpp，以便访问 CudaWeightPool 全量类型，
// 避免在头文件引入 CUDA backend 细节。
