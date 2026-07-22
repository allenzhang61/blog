#include "cpu_ops.hpp"

namespace llm {
namespace cpu {

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape(), DType::Float32, x.device(), x.requires_grad());
    constexpr double k = 0.7978845608028654;
    for (int64_t i = 0; i < x.numel(); ++i) {
        double v = x.data()[i];
        out.data()[i] = 0.5 * v * (1.0 + std::tanh(k * (v + 0.044715 * v * v * v)));
    }
    if (x.requires_grad()) {
        out.node->parents = {x};
        out.node->backward_fn = [x, out]() mutable {
            constexpr double k = 0.7978845608028654;
            for (int64_t i = 0; i < x.numel(); ++i) {
                double v = x.data()[i];
                double u = k * (v + 0.044715 * v * v * v);
                double th = std::tanh(u);
                double du = k * (1.0 + 3.0 * 0.044715 * v * v);
                double g = 0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * du;
                x.grad()[i] += out.grad()[i] * g;
            }
        };
    }
    return out;
}

} // namespace cpu
} // namespace llm
