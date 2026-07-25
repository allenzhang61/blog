#include "cpu_ops.hpp"

#include <numeric>
#include <stdexcept>

namespace llm {
namespace cpu {

Tensor add(const Tensor& a, const Tensor& b) {
    bool same_shape = a.shape() == b.shape();
    bool broadcast_batch = a.shape().size() == 3 && b.shape().size() == 2 &&
                           a.shape()[1] == b.shape()[0] && a.shape()[2] == b.shape()[1];
    bool broadcast_last = !a.shape().empty() && b.shape().size() == 1 && a.shape().back() == b.shape()[0];
    std::vector<int64_t> out_shape = a.shape();
    if (!same_shape && !broadcast_batch && !broadcast_last) {
        throw std::runtime_error("add shape mismatch");
    }
    Tensor out(out_shape, DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    for (int64_t i = 0; i < out.numel(); ++i) {
        double bv = (broadcast_batch || broadcast_last) ? b.data()[i % b.numel()] : b.data()[i];
        out.data()[i] = a.data()[i] + bv;
    }
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, broadcast_batch, broadcast_last]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    a.grad()[i] += out.grad()[i];
                }
            }
            if (b.requires_grad()) {
                if (broadcast_batch || broadcast_last) {
                    for (int64_t i = 0; i < out.numel(); ++i) {
                        b.grad()[i % b.numel()] += out.grad()[i];
                    }
                } else {
                    for (int64_t i = 0; i < out.numel(); ++i) {
                        b.grad()[i] += out.grad()[i];
                    }
                }
            }
        };
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor neg_b = Tensor::from_vector(b.data(), b.shape(), b.device(), b.requires_grad());
    for (auto& v : neg_b.data()) {
        v = -v;
    }
    if (b.requires_grad()) {
        neg_b.node->parents = {b};
        neg_b.node->backward_fn = [b, neg_b]() mutable {
            for (int64_t i = 0; i < neg_b.numel(); ++i) {
                b.grad()[i] -= neg_b.grad()[i];
            }
        };
    }
    return add(a, neg_b);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("mul shape mismatch");
    }
    Tensor out(a.shape(), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    for (int64_t i = 0; i < out.numel(); ++i) {
        out.data()[i] = a.data()[i] * b.data()[i];
    }
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    a.grad()[i] += b.data()[i] * out.grad()[i];
                }
            }
            if (b.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    b.grad()[i] += a.data()[i] * out.grad()[i];
                }
            }
        };
    }
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("div shape mismatch");
    }
    Tensor out(a.shape(), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    for (int64_t i = 0; i < out.numel(); ++i) {
        out.data()[i] = a.data()[i] / b.data()[i];
    }
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    a.grad()[i] += out.grad()[i] / b.data()[i];
                }
            }
            if (b.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    b.grad()[i] -= out.grad()[i] * a.data()[i] / (b.data()[i] * b.data()[i]);
                }
            }
        };
    }
    return out;
}

Tensor mul_scalar(const Tensor& a, double scalar) {
    Tensor out(a.shape(), DType::Float32, a.device(), a.requires_grad());
    for (int64_t i = 0; i < out.numel(); ++i) {
        out.data()[i] = a.data()[i] * scalar;
    }
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, scalar]() mutable {
            for (int64_t i = 0; i < out.numel(); ++i) {
                a.grad()[i] += out.grad()[i] * scalar;
            }
        };
    }
    return out;
}

Tensor pow(const Tensor& a, double exponent) {
    Tensor out(a.shape(), DType::Float32, a.device(), a.requires_grad());
    for (int64_t i = 0; i < out.numel(); ++i) {
        out.data()[i] = std::pow(a.data()[i], exponent);
    }
    if (out.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, exponent]() mutable {
            for (int64_t i = 0; i < out.numel(); ++i) {
                a.grad()[i] += exponent * std::pow(a.data()[i], exponent - 1.0) * out.grad()[i];
            }
        };
    }
    return out;
}

Tensor sum(const Tensor& a) {
    Tensor out({}, DType::Float32, a.device(), a.requires_grad());
    out.data()[0] = std::accumulate(a.data().begin(), a.data().end(), 0.0);
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            for (int64_t i = 0; i < a.numel(); ++i) {
                a.grad()[i] += out.grad()[0];
            }
        };
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor out = sum(a);
    out.data()[0] /= static_cast<double>(a.numel());
    if (a.requires_grad()) {
        out.node->backward_fn = [a, out]() mutable {
            for (int64_t i = 0; i < a.numel(); ++i) {
                a.grad()[i] += out.grad()[0] / static_cast<double>(a.numel());
            }
        };
    }
    return out;
}

Tensor max(const Tensor& a) {
    Tensor out({}, DType::Float32, a.device(), false);
    out.data()[0] = *std::max_element(a.data().begin(), a.data().end());
    return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (product(new_shape) != a.numel()) {
        throw std::runtime_error("reshape numel mismatch");
    }
    Tensor out(new_shape, a.data(), a.dtype(), a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            for (int64_t i = 0; i < out.numel(); ++i) {
                a.grad()[i] += out.grad()[i];
            }
        };
    }
    return out;
}

Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
    auto shape = a.shape();
    int64_t rank = static_cast<int64_t>(shape.size());
    dim0 = canonical_dim(dim0, rank);
    dim1 = canonical_dim(dim1, rank);
    std::vector<int64_t> out_shape = shape;
    std::swap(out_shape[dim0], out_shape[dim1]);
    Tensor out(out_shape, a.dtype(), a.device(), a.requires_grad());
    auto in_strides = strides_for(shape);
    auto out_strides = strides_for(out_shape);
    for (int64_t flat = 0; flat < out.numel(); ++flat) {
        int64_t rem = flat;
        std::vector<int64_t> idx(rank);
        for (int64_t d = 0; d < rank; ++d) {
            idx[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        std::swap(idx[dim0], idx[dim1]);
        int64_t in_flat = 0;
        for (int64_t d = 0; d < rank; ++d) {
            in_flat += idx[d] * in_strides[d];
        }
        out.data()[flat] = a.data()[in_flat];
    }
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, dim0, dim1]() mutable {
            Tensor gout(out.shape(), out.grad(), DType::Float32, out.device(), false);
            Tensor back = transpose(gout, dim0, dim1);
            for (int64_t i = 0; i < a.numel(); ++i) {
                a.grad()[i] += back.data()[i];
            }
        };
    }
    return out;
}

Tensor causal_mask(const Tensor& scores, int64_t sequence_length, double mask_value) {
    if (scores.shape().size() != 4 || scores.shape()[2] != sequence_length || scores.shape()[3] != sequence_length) {
        throw std::runtime_error("causal_mask expects scores [B,H,T,T]");
    }
    Tensor out(scores.shape(), scores.data(), scores.dtype(), scores.device(), scores.requires_grad());
    int64_t B = scores.shape()[0];
    int64_t H = scores.shape()[1];
    int64_t T = sequence_length;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < T; ++i) {
                for (int64_t j = i + 1; j < T; ++j) {
                    out.data()[((b * H + h) * T + i) * T + j] = mask_value;
                }
            }
        }
    }
    if (scores.requires_grad()) {
        out.node->parents = {scores};
        out.node->backward_fn = [scores, out, B, H, T]() mutable {
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t i = 0; i < T; ++i) {
                        for (int64_t j = 0; j <= i; ++j) {
                            int64_t idx = ((b * H + h) * T + i) * T + j;
                            scores.grad()[idx] += out.grad()[idx];
                        }
                    }
                }
            }
        };
    }
    return out;
}

} // namespace cpu
} // namespace llm
