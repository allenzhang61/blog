//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_RMSNORM_H
#define LOCAL_LLM_RMSNORM_H

#include <cstddef>

#include "llm/qwen/QwenWeights.h"
#include "Module.h"

class CudaWeightPool;

// RMSNorm（Qwen 使用，无 bias）：对最后一维做 RMS 归一化后乘以可学习权重。
//   y = x / sqrt(mean(x^2) + eps) * weight
// 权重形状 [hidden_size]，逐通道缩放。
class RMSNorm : public Module {
public:
    // weight：该 norm 的 gamma 权重（WeightData 引用）；pool：device 权重缓存；eps：数值稳定项。
    // one_plus：Qwen 的 RMSNorm 权重按 (1 + gamma) 缩放，默认 true。
    RMSNorm(const WeightData &weight, CudaWeightPool *pool, float eps, bool one_plus = true);

    // 原位或写出归一化结果。d_in / d_out 均为 device float 指针，
    // 形状 [rows, hidden_size]（prefill 时 rows=tokens，decode 时 rows=1）。
    // 允许 d_in == d_out 做原位归一化。
    void forward(const float *d_in, float *d_out, size_t rows, int hidden_size);

private:
    const WeightData &weight_;
    CudaWeightPool *pool_ = nullptr;
    float eps_ = 1e-6f;
    bool one_plus_ = true;
};


#endif //LOCAL_LLM_RMSNORM_H
