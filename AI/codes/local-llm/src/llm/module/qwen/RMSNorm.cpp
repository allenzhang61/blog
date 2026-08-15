//
// Created by zhangyoulun on 9/8/2026.
//

#include "RMSNorm.h"

#include <cstddef>
#include <stdexcept>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenForwardScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

namespace {
// DType -> kernel weight_type：0=bf16，1=f16，2=f32。
int weight_type_of(DType dtype) {
    if (dtype == DType::BF16) return 0;
    if (dtype == DType::F16) return 1;
    if (dtype == DType::F32) return 2;
    throw std::runtime_error(std::string("RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
}
} // namespace

RMSNorm::RMSNorm(const TensorView &weight, CudaWeightPool *pool, float eps, bool one_plus)
    : weight_(weight), pool_(pool), eps_(eps), one_plus_(one_plus) {}

void RMSNorm::forward(const float *d_in, float *d_out, size_t rows, int hidden_size,
                      QwenForwardScratch &scratch) {
    CudaWeight *w = pool_->cached_weight(weight_);
    if (!w) {
        throw std::runtime_error("RMSNorm 权重上传失败：" + weight_.name);
    }
    CudaWeight weight = w->try_dequant();
    const int wtype = weight_type_of(weight.dtype);
    launch_rms_norm(d_in, weight.ptr, wtype, d_out, static_cast<int>(rows), hidden_size, eps_,
                    /*one_plus=*/one_plus_, /*stream=*/nullptr);
}
