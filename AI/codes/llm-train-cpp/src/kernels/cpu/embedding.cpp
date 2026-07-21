#include "llm/cpu_ops.hpp"

namespace llm {
namespace cpu {

Tensor embedding(const Tensor& ids, const Tensor& weight) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("embedding weight must be 2D");
    }
    int64_t dim = weight.shape()[1];
    auto out_shape = ids.shape();
    out_shape.push_back(dim);
    Tensor out(out_shape, DType::Float32, weight.device(), weight.requires_grad());
    for (int64_t i = 0; i < ids.numel(); ++i) {
        int64_t id = static_cast<int64_t>(ids.data()[i]);
        for (int64_t d = 0; d < dim; ++d) {
            out.data()[i * dim + d] = weight.data()[id * dim + d];
        }
    }
    if (weight.requires_grad()) {
        out.node->parents = {weight};
        out.node->backward_fn = [ids, weight, out, dim]() mutable {
            for (int64_t i = 0; i < ids.numel(); ++i) {
                int64_t id = static_cast<int64_t>(ids.data()[i]);
                for (int64_t d = 0; d < dim; ++d) {
                    weight.grad()[id * dim + d] += out.grad()[i * dim + d];
                }
            }
        };
    }
    return out;
}

} // namespace cpu
} // namespace llm
