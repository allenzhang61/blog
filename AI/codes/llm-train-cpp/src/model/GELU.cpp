#include "llm/model/GELU.hpp"

#include "llm/ops.hpp"

namespace llm {

// 逐元素激活，形状不变：x: (...) -> 返回: (...)（同形状）
Tensor GELU::forward(const Tensor& x) {
    return ops::gelu(x);
}

} // namespace llm
