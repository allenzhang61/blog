#include "cpu_ops.hpp"

namespace llm {
namespace cpu {

Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    int64_t C = x.shape().back();
    int64_t rows = x.numel() / C;
    Tensor out(x.shape(), DType::Float32, x.device(), x.requires_grad() || scale.requires_grad() || shift.requires_grad());
    std::vector<double> xhat(x.numel(), 0.0);
    std::vector<double> invs(rows, 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            mean += x.data()[r * C + c];
        }
        mean /= C;
        double var = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            double z = x.data()[r * C + c] - mean;
            var += z * z;
        }
        var /= C;
        double inv = 1.0 / std::sqrt(var + eps);
        invs[r] = inv;
        for (int64_t c = 0; c < C; ++c) {
            int64_t idx = r * C + c;
            xhat[idx] = (x.data()[idx] - mean) * inv;
            out.data()[idx] = xhat[idx] * scale.data()[c] + shift.data()[c];
        }
    }
    if (out.requires_grad()) {
        out.node->parents = {x, scale, shift};
        out.node->backward_fn = [x, scale, shift, out, C, rows, xhat, invs]() mutable {
            if (scale.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    scale.grad()[i % C] += out.grad()[i] * xhat[i];
                }
            }
            if (shift.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    shift.grad()[i % C] += out.grad()[i];
                }
            }
            if (x.requires_grad()) {
                for (int64_t r = 0; r < rows; ++r) {
                    double sum_dxhat = 0.0;
                    double sum_dxhat_xhat = 0.0;
                    for (int64_t c = 0; c < C; ++c) {
                        int64_t idx = r * C + c;
                        double dxhat = out.grad()[idx] * scale.data()[c];
                        sum_dxhat += dxhat;
                        sum_dxhat_xhat += dxhat * xhat[idx];
                    }
                    for (int64_t c = 0; c < C; ++c) {
                        int64_t idx = r * C + c;
                        double dxhat = out.grad()[idx] * scale.data()[c];
                        x.grad()[idx] += (static_cast<double>(C) * dxhat - sum_dxhat - xhat[idx] * sum_dxhat_xhat) *
                                         invs[r] / static_cast<double>(C);
                    }
                }
            }
        };
    }
    return out;
}

} // namespace cpu
} // namespace llm
