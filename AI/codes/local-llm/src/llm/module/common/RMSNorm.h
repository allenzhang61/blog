//
// Created by zhangyoulun on 16/8/2026.
//

#ifndef LOCAL_LLM_COMMON_RMSNORM_H
#define LOCAL_LLM_COMMON_RMSNORM_H

#include <cstddef>

#include "format/MF.h"

class CudaWeightPool;

// RMSNorm（无 bias）：对最后一维做 RMS 归一化后乘以可学习权重。
//   y = x / sqrt(mean(x^2) + eps) * weight
// 权重形状 [hidden_size]，逐通道缩放。
//
// 无状态工具，直接调用静态 forward，权重、eps、one_plus 均在调用时传入：
//   - Qwen 风格：one_plus 传 true（权重按 (1 + gamma) 缩放）。
//   - DeepSeek 风格：one_plus 传 false（权重按原值缩放）。
class RMSNorm {
public:
    // 对 d_input 做归一化写入 d_output。d_input / d_output 均为 device float 指针，
    // 形状 [rows, hidden_size]（prefill 时 rows=tokens，decode 时 rows=1）。
    // 允许 d_input == d_output 做原位归一化。
    // pool：device 权重缓存；weight：gamma 权重；eps：数值稳定项；
    // one_plus：权重是否按 (1 + gamma) 缩放。
    static void forward(CudaWeightPool *pool, const MFTensorView &weight,
                        const float *d_input, float *d_output,
                        size_t rows, int hidden_size, float eps, bool one_plus);
};

#endif // LOCAL_LLM_COMMON_RMSNORM_H
