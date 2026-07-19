#pragma once

#include "llm/tensor.hpp"

namespace llm::ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor mul_scalar(const Tensor& a, double scalar);
Tensor pow(const Tensor& a, double exponent);
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor max(const Tensor& a);
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape);
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor batch_matmul(const Tensor& a, const Tensor& b);
Tensor softmax(const Tensor& a, int64_t dim = -1);
Tensor log_softmax(const Tensor& a, int64_t dim = -1);
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor embedding(const Tensor& ids, const Tensor& weight);
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps = 1e-5);
Tensor gelu(const Tensor& x);

} // namespace llm::ops

namespace llm {

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);

} // namespace llm
