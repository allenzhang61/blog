#include "llm/ops.hpp"

namespace llm {

Tensor operator+(const Tensor& a, const Tensor& b) { return ops::add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return ops::sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return ops::mul(a, b); }

} // namespace llm
