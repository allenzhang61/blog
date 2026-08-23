//
// Created by zhangyoulun on 8/8/2026.
//

#include "CudaWeight.h"

#include "backend/cuda/mem/CudaWeightDequantPool.h"
#include "backend/cuda/mem/Quant.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "../common.h"

CudaWeight::CudaWeight(size_t bytes, cudaDataType_t type, bool zero, const std::string &what)
    : bytes(bytes), type(type) {
    // 由 cuBLAS 数据类型推导原始 DType，使基于本缓冲区的 GPUTensor 携带正确 dtype（如 recurrent state bf16）。
    if (type == CUDA_R_16BF) dtype = DType::BF16;
    else if (type == CUDA_R_16F) dtype = DType::F16;
    else dtype = DType::F32;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc " + what + " 失败");
    if (zero) {
        check_cuda(cudaMemset(ptr, 0, bytes), "cudaMemset " + what + " 失败");
    }
}

CudaWeight::~CudaWeight() {
    reset();
}

CudaWeight CudaWeight::make_view(void *ptr, size_t bytes, cudaDataType_t type,
                                 DType dtype, int64_t num_elements, std::string name,
                                 std::shared_ptr<void> keep_alive) {
    CudaWeight w;
    w.ptr = ptr;
    w.bytes = bytes;
    w.type = type;
    w.dtype = dtype;
    w.num_elements = num_elements;
    w.name = std::move(name);
    w.keep_alive_ = std::move(keep_alive);
    w.owns_ = false; // 视图不拥有内存，析构不释放。
    return w;
}

CudaWeight::CudaWeight(CudaWeight &&other) noexcept
    : ptr(other.ptr),
      bytes(other.bytes),
      type(other.type),
      dtype(other.dtype),
      num_elements(other.num_elements),
      name(std::move(other.name)),
      keep_alive_(std::move(other.keep_alive_)),
      owns_(other.owns_) {
    other.ptr = nullptr;
    other.bytes = 0;
    other.num_elements = 0;
    other.owns_ = true;
}

CudaWeight &CudaWeight::operator=(CudaWeight &&other) noexcept {
    if (this != &other) {
        reset();
        ptr = other.ptr;
        bytes = other.bytes;
        type = other.type;
        dtype = other.dtype;
        num_elements = other.num_elements;
        name = std::move(other.name);
        keep_alive_ = std::move(other.keep_alive_);
        owns_ = other.owns_;
        other.ptr = nullptr;
        other.bytes = 0;
        other.num_elements = 0;
        other.owns_ = true;
    }
    return *this;
}

CudaWeight CudaWeight::try_dequant() const {
    if (!Quant::is_quantized_dtype(dtype)) {
        return make_view(ptr, bytes, type, dtype, num_elements, name);
    }
    CudaWeightDequantPool *pool = global_cuda_weight_dequant_pool();
    if (pool != nullptr) {
        return pool->cached_dequant(*this);
    }

    throw std::runtime_error("量化权重缺少全局 dequant pool: " + name);
}

void CudaWeight::reset() {
    if (ptr && owns_) {
        cudaFree(ptr);
    }
    ptr = nullptr;
    bytes = 0;
    num_elements = 0;
    name.clear();
    keep_alive_.reset();
    owns_ = true;
}
