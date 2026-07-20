#pragma once

#include "llm/model/Module.hpp"

namespace llm {

// 全连接线性层。
// 对最后一维做 y = xW + b。
class Linear : public Module {
public:
    // 权重矩阵，形状通常是 [in_features, out_features]。
    Tensor weight;

    // 偏置向量，形状通常是 [out_features]。
    Tensor bias;

    // 是否启用 bias。
    bool use_bias{true};

    // 创建线性层。
    // bias_enabled=false 时只使用权重矩阵。
    Linear(int64_t in_features, int64_t out_features, bool bias_enabled = true, Device device = {});

    // 对输入最后一维做线性投影，前面的 batch/seq 维度保持不变。
    Tensor forward(const Tensor& x);

    // 返回 weight 和可选 bias。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
