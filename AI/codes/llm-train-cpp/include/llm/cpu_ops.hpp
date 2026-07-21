#pragma once

#include "llm/tensor.hpp"

namespace llm::cpu {

// CPU 版本的逐元素加法，支持必要的简单广播。
Tensor add(const Tensor& a, const Tensor& b);

// CPU 版本的逐元素减法。
Tensor sub(const Tensor& a, const Tensor& b);

// CPU 版本的逐元素乘法。
Tensor mul(const Tensor& a, const Tensor& b);

// CPU 版本的逐元素除法。
Tensor div(const Tensor& a, const Tensor& b);

// CPU 版本的张量乘标量。
Tensor mul_scalar(const Tensor& a, double scalar);

// CPU 版本的逐元素幂运算。
Tensor pow(const Tensor& a, double exponent);

// CPU 版本的求和，返回标量张量。
Tensor sum(const Tensor& a);

// CPU 版本的求均值，返回标量张量。
Tensor mean(const Tensor& a);

// CPU 版本的求最大值，返回标量张量。
Tensor max(const Tensor& a);

// CPU 版本的 reshape。
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape);

// CPU 版本的维度交换。
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);

// CPU 版本的矩阵乘法。
Tensor matmul(const Tensor& a, const Tensor& b);

// CPU 版本的批量矩阵乘法。
Tensor batch_matmul(const Tensor& a, const Tensor& b);

// CPU 版本的 softmax。
Tensor softmax(const Tensor& a, int64_t dim = -1);

// CPU 版本的 log_softmax。
Tensor log_softmax(const Tensor& a, int64_t dim = -1);

// CPU 版本的交叉熵损失。
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

// CPU 版本的 embedding 查表。
Tensor embedding(const Tensor& ids, const Tensor& weight);

// CPU 版本的 LayerNorm。
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps = 1e-5);

// CPU 版本的 GELU。
Tensor gelu(const Tensor& x);

} // namespace llm::cpu
