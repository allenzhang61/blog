//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/RMSNorm.h"

#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

#include <cstddef>
#include <stdexcept>

namespace {
// DType -> kernel weight_type：0=bf16，1=f16，2=f32。
int weight_type_of(DType dtype) {
    if (dtype == DType::BF16) return 0;
    if (dtype == DType::F16) return 1;
    if (dtype == DType::F32) return 2;
    throw std::runtime_error(std::string("RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
}
} // namespace

void RMSNorm::forward(CudaWeightPool *pool, const MFTensorView &weight,
                      const float *d_input, float *d_output,
                      size_t rows, int hidden_size, float eps, bool one_plus) {
    CudaWeight w = pool->cached_weight(weight)->try_dequant();
    launch_rms_norm(d_input, d_output, w.ptr, weight_type_of(w.dtype),
                    static_cast<int>(rows), hidden_size, eps, one_plus, nullptr);
}
