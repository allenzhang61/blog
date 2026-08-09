//
// Created by zhangyoulun on 9/8/2026.
//

#include "RMSNorm.h"

#include <stdexcept>

#include "llm/qwen/QwenWeights.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

namespace {
// safetensors dtype -> kernel weight_type：0=bf16，1=f16，2=f32。
int weight_type_of(const std::string &dtype) {
    if (dtype == "BF16") return 0;
    if (dtype == "F16") return 1;
    if (dtype == "F32") return 2;
    throw std::runtime_error("RMSNorm 不支持的 norm dtype：" + dtype);
}
} // namespace

RMSNorm::RMSNorm(const WeightData &weight, CudaWeightPool *pool, float eps, bool one_plus)
    : weight_(weight), pool_(pool), eps_(eps), one_plus_(one_plus) {}

void RMSNorm::forward(const float *d_in, float *d_out, int rows, int hidden_size) {
    CudaWeight *w = pool_->cached_weight(weight_);
    if (!w) {
        throw std::runtime_error("RMSNorm 权重上传失败：" + weight_.info->name);
    }
    const int wtype = weight_type_of(weight_.info->dtype);
    launch_rms_norm(d_in, w->ptr, wtype, d_out, rows, hidden_size, eps_,
                    /*one_plus=*/one_plus_, /*stream=*/nullptr);
}
