#pragma once

#include "llm/tensor.hpp"

namespace llm::ops {

// 逐元素加法，支持必要的简单广播。
Tensor add(const Tensor& a, const Tensor& b);

// 逐元素减法。
Tensor sub(const Tensor& a, const Tensor& b);

// 逐元素乘法。
Tensor mul(const Tensor& a, const Tensor& b);

// 逐元素除法。
Tensor div(const Tensor& a, const Tensor& b);

// 张量乘以标量。
Tensor mul_scalar(const Tensor& a, double scalar);

// 张量逐元素幂运算。
Tensor pow(const Tensor& a, double exponent);

// 对所有元素求和，返回标量张量。
Tensor sum(const Tensor& a);

// 对所有元素求均值，返回标量张量。
Tensor mean(const Tensor& a);

// 对所有元素取最大值，返回标量张量。
Tensor max(const Tensor& a);

// 改变张量形状，不改变元素顺序。
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape);

// 交换两个维度。
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);

// 矩阵乘法，主要用于 Linear 和注意力分数计算。
Tensor matmul(const Tensor& a, const Tensor& b);

// 批量矩阵乘法，用于多头注意力中的每个 batch/head。
Tensor batch_matmul(const Tensor& a, const Tensor& b);

// 对注意力分数施加 causal mask，把每个 [T,T] 矩阵的上三角未来位置写成 mask_value。
Tensor causal_mask(const Tensor& scores, int64_t sequence_length, double mask_value = -1e9);

// 在指定维度上做 softmax，常用于注意力权重。
Tensor softmax(const Tensor& a, int64_t dim = -1);

// 在指定维度上做 log_softmax。
Tensor log_softmax(const Tensor& a, int64_t dim = -1);

// 交叉熵损失。logits 形状通常是 [N, vocab_size]，targets 是 [N]。
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

// embedding 查表：ids 中每个 token id 对应 weight 的一行。
Tensor embedding(const Tensor& ids, const Tensor& weight);

// LayerNorm 运算，最后一维按 scale/shift 做归一化和仿射变换。
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps = 1e-5);

// GELU 激活函数。
Tensor gelu(const Tensor& x);

} // namespace llm::ops
