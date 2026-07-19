#pragma once

#include "llm/tensor.hpp"

namespace llm::metal {

bool available();
std::string status();

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul_scalar(const Tensor& a, double scalar);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor batch_matmul(const Tensor& a, const Tensor& b);
Tensor softmax(const Tensor& a, int64_t dim = -1);
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor embedding(const Tensor& ids, const Tensor& weight);
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps = 1e-5);
Tensor gelu(const Tensor& x);

} // namespace llm::metal
