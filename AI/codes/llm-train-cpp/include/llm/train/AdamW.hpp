#pragma once

#include "llm/tensor.hpp"

namespace llm {

// AdamW 优化器。
// 负责根据参数梯度更新模型权重，并支持 decoupled weight decay。
class AdamW {
public:
    // params 是待优化的参数指针列表。
    explicit AdamW(std::vector<Tensor*> params, double lr = 1e-3, double weight_decay = 0.0,
                   double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

    // 清空所有参数梯度。
    void zero_grad();

    // 执行一次 AdamW 参数更新。
    void step();

private:
    // 待优化参数。
    std::vector<Tensor*> params_;

    // Adam 一阶矩估计。
    std::vector<std::vector<double>> m_;

    // Adam 二阶矩估计。
    std::vector<std::vector<double>> v_;

    // CUDA 参数的一阶矩 device buffer；CPU 参数保持为空。
    std::vector<std::shared_ptr<TensorCudaStorage>> cuda_m_;

    // CUDA 参数的二阶矩 device buffer；CPU 参数保持为空。
    std::vector<std::shared_ptr<TensorCudaStorage>> cuda_v_;

    // 学习率。
    double lr_;

    // 权重衰减系数。
    double weight_decay_;

    // 一阶矩衰减系数。
    double beta1_;

    // 二阶矩衰减系数。
    double beta2_;

    // 数值稳定项。
    double eps_;

    // 当前优化步数。
    int64_t step_{0};
};

} // namespace llm
