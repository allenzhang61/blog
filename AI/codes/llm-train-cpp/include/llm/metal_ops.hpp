#pragma once

#include "llm/tensor.hpp"

namespace llm::metal {

// 当前运行环境是否可使用 Metal。
bool available();

// Metal 后端状态说明。
std::string status();

// Metal 版本的逐元素加法。
Tensor add(const Tensor& a, const Tensor& b);

// Metal 版本的逐元素乘法。
Tensor mul(const Tensor& a, const Tensor& b);

// Metal 版本的张量乘标量。
Tensor mul_scalar(const Tensor& a, double scalar);

// Metal 版本的矩阵乘法。
Tensor matmul(const Tensor& a, const Tensor& b);

// Metal 版本的批量矩阵乘法。
Tensor batch_matmul(const Tensor& a, const Tensor& b);

// Metal 版本的 softmax。
Tensor softmax(const Tensor& a, int64_t dim = -1);

// Metal 版本的交叉熵损失。
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

// Metal 版本的 embedding 查表。
Tensor embedding(const Tensor& ids, const Tensor& weight);

// Metal 版本的 LayerNorm。
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps = 1e-5);

// Metal 版本的 GELU。
Tensor gelu(const Tensor& x);

} // namespace llm::metal
