#include "llm/ops.hpp"

namespace llm {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
    ensure_cpu(a); ensure_cpu(b);
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul expects [m,k] x [k,n]");
    }
    int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    Tensor out({m, n}, DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int64_t p = 0; p < k; ++p) acc += a.data()[i * k + p] * b.data()[p * n + j];
            out.data()[i * n + j] = acc;
        }
    }
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, m, k, n]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < m; ++i)
                    for (int64_t p = 0; p < k; ++p)
                        for (int64_t j = 0; j < n; ++j)
                            a.grad()[i * k + p] += out.grad()[i * n + j] * b.data()[p * n + j];
            }
            if (b.requires_grad()) {
                for (int64_t p = 0; p < k; ++p)
                    for (int64_t j = 0; j < n; ++j)
                        for (int64_t i = 0; i < m; ++i)
                            b.grad()[p * n + j] += a.data()[i * k + p] * out.grad()[i * n + j];
            }
        };
    }
    return out;
}

Tensor batch_matmul(const Tensor& a, const Tensor& b) {
    ensure_cpu(a); ensure_cpu(b);
    if (a.shape().size() != 4 || b.shape().size() != 4) throw std::runtime_error("batch_matmul expects 4D tensors");
    int64_t B = a.shape()[0], H = a.shape()[1], M = a.shape()[2], K = a.shape()[3];
    if (b.shape()[0] != B || b.shape()[1] != H || b.shape()[2] != K) throw std::runtime_error("batch_matmul shape mismatch");
    int64_t N = b.shape()[3];
    Tensor out({B, H, M, N}, DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    for (int64_t bb = 0; bb < B; ++bb)
        for (int64_t hh = 0; hh < H; ++hh)
            for (int64_t i = 0; i < M; ++i)
                for (int64_t j = 0; j < N; ++j) {
                    double acc = 0.0;
                    for (int64_t p = 0; p < K; ++p) {
                        int64_t ai = ((bb * H + hh) * M + i) * K + p;
                        int64_t bi = ((bb * H + hh) * K + p) * N + j;
                        acc += a.data()[ai] * b.data()[bi];
                    }
                    out.data()[((bb * H + hh) * M + i) * N + j] = acc;
                }
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

} // namespace ops
} // namespace llm
