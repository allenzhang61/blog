//
// Created by zhangyoulun on 16/8/2026.
//

#ifndef LOCAL_LLM_COMMON_RMSNORM_H
#define LOCAL_LLM_COMMON_RMSNORM_H

#include "tensor/StorageTensor.h"

class GPUTensor;

// RMSNorm（无 bias）：对最后一维做 RMS 归一化后乘以可学习权重。
//   y = x / sqrt(mean(x^2) + eps) * s_weight
// 权重形状 [hidden_size]，逐通道缩放。
//
// 无状态工具，调用时显式传入权重：
//   - Qwen 风格：one_plus 传 true（权重按 (1 + gamma) 缩放）。
//   - DeepSeek 风格：one_plus 传 false（权重按原值缩放）。
class RMSNorm {
public:
    // 对 g_input 做归一化写入 g_output。g_input / g_output 均为 device 激活视图（GPUTensor），
    // 形状 [rows, hidden_size]（prefill 时 rows=tokens，decode 时 rows=1），rows/hidden_size
    // 由 g_input.shape 推出。允许 g_input 与 g_output 指向同一 device 内存做原位归一化。
    // s_weight：gamma 权重；eps：数值稳定项；
    // one_plus：权重是否按 (1 + gamma) 缩放。
    static void forward(const StorageTensor &s_weight, const GPUTensor &g_input_f32,
                        const GPUTensor &g_output_f32, float eps, bool one_plus);
};

#endif // LOCAL_LLM_COMMON_RMSNORM_H
