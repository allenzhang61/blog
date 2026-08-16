//
// Created by zhangyoulun on 15/8/2026.
//

#include "RMSNorm.h"

#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

#include <stdexcept>

namespace {

// DType -> kernel weight_type：0=bf16，1=f16，2=f32。
int weight_type_of(DType dtype) {
    if (dtype == DType::BF16) return 0;
    if (dtype == DType::F16) return 1;
    if (dtype == DType::F32) return 2;
    throw std::runtime_error(std::string("DeepSeek RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
}

} // namespace

namespace deepseek {

RMSNorm::RMSNorm(CudaWeightPool *pool, float eps)
    : pool_(pool), eps_(eps) {}

void RMSNorm::forward(const MFTensorView &weight, const float *d_in, float *d_out,
                      size_t rows, int hidden_size) {
    CudaWeight *resident = pool_->cached_weight(weight);
    if (!resident) {
        throw std::runtime_error("DeepSeek RMSNorm 权重上传失败：" + weight.name);
    }
    CudaWeight w = resident->try_dequant();
    launch_rms_norm(d_in, w.ptr, weight_type_of(w.dtype), d_out,
                    static_cast<int>(rows), hidden_size, eps_, false, nullptr);
}

} // namespace deepseek
