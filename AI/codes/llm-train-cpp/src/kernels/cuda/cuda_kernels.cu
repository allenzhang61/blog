#include "cuda_ops.hpp"
#include "cuda_runtime.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

// 该文件是 CUDA 后端的真实实现，结构与 Metal 后端一一对应：
// - 前向计算在 GPU 上跑真实 __global__ kernel（对应 metal_kernels.metal）；
// - host 端调度/内存搬运封装在 cuda_runtime.cu 的 CudaRuntime 中（对应 MetalRuntime）；
// - 反向传播沿用 host mirror（与 MetalKernels.mm 的 backward 完全一致）。
//
// 说明：本机（Apple M3）无 NVIDIA GPU 与 CUDA Toolkit，无法在此环境编译验证，
// 逻辑严格镜像已验证通过的 Metal 实现。

namespace {

std::vector<float> to_float(const std::vector<double>& values) {
    return std::vector<float>(values.begin(), values.end());
}

std::vector<double> to_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

} // namespace

namespace llm::cuda {

using detail::CudaRuntime;

bool available() {
    return CudaRuntime::instance().available();
}

std::string status() {
    return CudaRuntime::instance().status();
}

Tensor add(const Tensor& a, const Tensor& b) {
    bool same_shape = a.shape() == b.shape();
    bool broadcast_batch = a.shape().size() == 3 && b.shape().size() == 2 &&
                           a.shape()[1] == b.shape()[0] && a.shape()[2] == b.shape()[1];
    bool broadcast_last = !a.shape().empty() && b.shape().size() == 1 && a.shape().back() == b.shape()[0];
    if (!same_shape && !broadcast_batch && !broadcast_last) {
        throw std::runtime_error("CUDA add shape mismatch");
    }
    auto out_data = CudaRuntime::instance().elementwise2("add", to_float(a.data()), to_float(b.data()),
                                                         static_cast<unsigned int>(b.numel()));
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
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

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("CUDA mul expects same shape");
    }
    auto out_data = CudaRuntime::instance().elementwise2("mul", to_float(a.data()), to_float(b.data()), 0);
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
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

Tensor mul_scalar(const Tensor& a, double scalar) {
    auto out_data = CudaRuntime::instance().mul_scalar(to_float(a.data()), static_cast<float>(scalar));
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
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

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("CUDA div expects same shape");
    }
    auto out_data = CudaRuntime::instance().elementwise2("div", to_float(a.data()), to_float(b.data()), 0);
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
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

Tensor sub(const Tensor& a, const Tensor& b) {
    auto neg_data = CudaRuntime::instance().unary("neg", to_float(b.data()));
    Tensor neg_b(b.shape(), to_double(neg_data), DType::Float32, b.device(), b.requires_grad());
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

Tensor pow(const Tensor& a, double exponent) {
    auto out_data = CudaRuntime::instance().unary("pow", to_float(a.data()), static_cast<float>(exponent));
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
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
    float value = CudaRuntime::instance().reduce("sum", to_float(a.data()));
    Tensor out({}, DType::Float32, a.device(), a.requires_grad());
    out.data()[0] = value;
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
    float value = CudaRuntime::instance().reduce("max", to_float(a.data()));
    Tensor out({}, DType::Float32, a.device(), false);
    out.data()[0] = value;
    return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (product(new_shape) != a.numel()) {
        throw std::runtime_error("CUDA reshape numel mismatch");
    }
    auto out_data = CudaRuntime::instance().unary("copy", to_float(a.data()));
    Tensor out(new_shape, to_double(out_data), DType::Float32, a.device(), a.requires_grad());
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
    auto in_strides = strides_for(shape);
    auto out_strides = strides_for(out_shape);
    // host 端预计算「输出扁平位置 -> 输入扁平位置」映射，交给 gather_kernel 在 GPU 上重排。
    std::vector<unsigned int> index(static_cast<size_t>(a.numel()));
    for (int64_t flat = 0; flat < a.numel(); ++flat) {
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
        index[flat] = static_cast<unsigned int>(in_flat);
    }
    auto out_data = CudaRuntime::instance().gather(to_float(a.data()), index);
    Tensor out(out_shape, to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, index]() mutable {
            for (int64_t flat = 0; flat < out.numel(); ++flat) {
                a.grad()[index[flat]] += out.grad()[flat];
            }
        };
    }
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("CUDA matmul expects [m,k] x [k,n]");
    }
    unsigned int m = static_cast<unsigned int>(a.shape()[0]);
    unsigned int k = static_cast<unsigned int>(a.shape()[1]);
    unsigned int n = static_cast<unsigned int>(b.shape()[1]);
    auto out_data = CudaRuntime::instance().matmul(to_float(a.data()), to_float(b.data()), m, k, n);
    Tensor out({static_cast<int64_t>(m), static_cast<int64_t>(n)}, to_double(out_data), DType::Float32,
               a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, m, k, n]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
                    for (int64_t p = 0; p < static_cast<int64_t>(k); ++p) {
                        for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
                            a.grad()[i * k + p] += out.grad()[i * n + j] * b.data()[p * n + j];
                        }
                    }
                }
            }
            if (b.requires_grad()) {
                for (int64_t p = 0; p < static_cast<int64_t>(k); ++p) {
                    for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
                        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
                            b.grad()[p * n + j] += a.data()[i * k + p] * out.grad()[i * n + j];
                        }
                    }
                }
            }
        };
    }
    return out;
}

Tensor batch_matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 4 || b.shape().size() != 4) {
        throw std::runtime_error("CUDA batch_matmul expects 4D tensors");
    }
    unsigned int B = static_cast<unsigned int>(a.shape()[0]);
    unsigned int H = static_cast<unsigned int>(a.shape()[1]);
    unsigned int M = static_cast<unsigned int>(a.shape()[2]);
    unsigned int K = static_cast<unsigned int>(a.shape()[3]);
    if (b.shape()[0] != B || b.shape()[1] != H || b.shape()[2] != K) {
        throw std::runtime_error("CUDA batch_matmul shape mismatch");
    }
    unsigned int N = static_cast<unsigned int>(b.shape()[3]);
    auto out_data = CudaRuntime::instance().batch_matmul(to_float(a.data()), to_float(b.data()), B, H, M, K, N);
    Tensor out({B, H, M, N}, to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, B, H, M, K, N]() mutable {
            if (a.requires_grad()) {
                for (int64_t bb = 0; bb < B; ++bb)
                    for (int64_t hh = 0; hh < H; ++hh)
                        for (int64_t i = 0; i < M; ++i)
                            for (int64_t p = 0; p < K; ++p)
                                for (int64_t j = 0; j < N; ++j) {
                                    int64_t ai = ((bb * H + hh) * M + i) * K + p;
                                    int64_t bi = ((bb * H + hh) * K + p) * N + j;
                                    int64_t oi = ((bb * H + hh) * M + i) * N + j;
                                    a.grad()[ai] += out.grad()[oi] * b.data()[bi];
                                }
            }
            if (b.requires_grad()) {
                for (int64_t bb = 0; bb < B; ++bb)
                    for (int64_t hh = 0; hh < H; ++hh)
                        for (int64_t p = 0; p < K; ++p)
                            for (int64_t j = 0; j < N; ++j)
                                for (int64_t i = 0; i < M; ++i) {
                                    int64_t ai = ((bb * H + hh) * M + i) * K + p;
                                    int64_t bi = ((bb * H + hh) * K + p) * N + j;
                                    int64_t oi = ((bb * H + hh) * M + i) * N + j;
                                    b.grad()[bi] += a.data()[ai] * out.grad()[oi];
                                }
            }
        };
    }
    return out;
}

Tensor softmax(const Tensor& a, int64_t dim) {
    int64_t rank = static_cast<int64_t>(a.shape().size());
    dim = canonical_dim(dim, rank);
    if (dim != rank - 1) {
        throw std::runtime_error("CUDA softmax currently supports last dim only");
    }
    unsigned int width = static_cast<unsigned int>(a.shape().back());
    unsigned int rows = static_cast<unsigned int>(a.numel() / width);
    auto out_data = CudaRuntime::instance().softmax(to_float(a.data()), rows, width);
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, rows, width]() mutable {
            for (int64_t r = 0; r < rows; ++r) {
                double dot = 0.0;
                for (int64_t c = 0; c < width; ++c) {
                    dot += out.grad()[r * width + c] * out.data()[r * width + c];
                }
                for (int64_t c = 0; c < width; ++c) {
                    a.grad()[r * width + c] += out.data()[r * width + c] * (out.grad()[r * width + c] - dot);
                }
            }
        };
    }
    return out;
}

Tensor log_softmax(const Tensor& a, int64_t dim) {
    int64_t rank = static_cast<int64_t>(a.shape().size());
    int64_t cdim = canonical_dim(dim, rank);
    if (cdim != rank - 1) {
        throw std::runtime_error("CUDA log_softmax currently supports last dim only");
    }
    unsigned int width = static_cast<unsigned int>(a.shape().back());
    unsigned int rows = static_cast<unsigned int>(a.numel() / width);
    auto probs = CudaRuntime::instance().softmax(to_float(a.data()), rows, width);
    auto out_data = CudaRuntime::instance().unary("log", probs);
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    if (logits.shape().size() != 3) {
        throw std::runtime_error("CUDA cross_entropy expects logits [B,T,V]");
    }
    int64_t B = logits.shape()[0], T = logits.shape()[1], V = logits.shape()[2];
    if (targets.numel() != B * T) {
        throw std::runtime_error("CUDA cross_entropy target shape mismatch");
    }

    auto row_losses = CudaRuntime::instance().cross_entropy_row_losses(to_float(logits.data()), to_float(targets.data()),
                                                                       static_cast<unsigned int>(B * T), static_cast<unsigned int>(V));
    double loss = 0.0;
    for (auto value : row_losses) {
        loss += value;
    }
    Tensor out({}, DType::Float32, logits.device(), logits.requires_grad());
    out.data()[0] = loss / static_cast<double>(B * T);

    std::vector<double> probs(logits.numel(), 0.0);
    for (int64_t row = 0; row < B * T; ++row) {
        double mx = -1e100;
        for (int64_t v = 0; v < V; ++v) {
            mx = std::max(mx, logits.data()[row * V + v]);
        }
        double denom = 0.0;
        for (int64_t v = 0; v < V; ++v) {
            probs[row * V + v] = std::exp(logits.data()[row * V + v] - mx);
            denom += probs[row * V + v];
        }
        for (int64_t v = 0; v < V; ++v) {
            probs[row * V + v] /= denom;
        }
    }
    if (logits.requires_grad()) {
        out.node->parents = {logits};
        out.node->backward_fn = [logits, targets, out, probs, B, T, V]() mutable {
            for (int64_t row = 0; row < B * T; ++row) {
                int64_t target = static_cast<int64_t>(targets.data()[row]);
                for (int64_t v = 0; v < V; ++v) {
                    double g = probs[row * V + v];
                    if (v == target) {
                        g -= 1.0;
                    }
                    logits.grad()[row * V + v] += out.grad()[0] * g / static_cast<double>(B * T);
                }
            }
        };
    }
    return out;
}

Tensor embedding(const Tensor& ids, const Tensor& weight) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("CUDA embedding weight must be 2D");
    }
    unsigned int count = static_cast<unsigned int>(ids.numel());
    unsigned int dim = static_cast<unsigned int>(weight.shape()[1]);
    auto out_shape = ids.shape();
    out_shape.push_back(dim);
    auto out_data = CudaRuntime::instance().embedding(to_float(ids.data()), to_float(weight.data()), count, dim);
    Tensor out(out_shape, to_double(out_data), DType::Float32, weight.device(), weight.requires_grad());
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

Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    unsigned int C = static_cast<unsigned int>(x.shape().back());
    unsigned int rows = static_cast<unsigned int>(x.numel() / C);
    auto out_data = CudaRuntime::instance().layernorm(to_float(x.data()), to_float(scale.data()), to_float(shift.data()),
                                                      rows, C, static_cast<float>(eps));
    Tensor out(x.shape(), to_double(out_data), DType::Float32, x.device(),
               x.requires_grad() || scale.requires_grad() || shift.requires_grad());

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
        invs[r] = 1.0 / std::sqrt(var + eps);
        for (int64_t c = 0; c < C; ++c) {
            xhat[r * C + c] = (x.data()[r * C + c] - mean) * invs[r];
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

Tensor gelu(const Tensor& x) {
    auto out_data = CudaRuntime::instance().unary("gelu", to_float(x.data()));
    Tensor out(x.shape(), to_double(out_data), DType::Float32, x.device(), x.requires_grad());
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

} // namespace llm::cuda
