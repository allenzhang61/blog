#include "llm/model/GELU.hpp"

#include "llm/ops.hpp"

namespace llm {

Tensor GELU::forward(const Tensor& x) {
    return ops::gelu(x);
}

} // namespace llm
