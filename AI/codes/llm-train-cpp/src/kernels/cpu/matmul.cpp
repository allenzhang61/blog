#include "cpu_ops.hpp"

#include <stdexcept>

#if defined(LLM_CPP_USE_BLAS)
// 使用 Accelerate 新版 CBLAS 接口（避免 macOS 13.3+ 的 deprecated 警告）。
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace llm {
namespace cpu {

namespace {

// C = A[m,k] * B[k,n]（行主序），beta=0 覆盖，beta=1 累加。
inline void gemm_nn(int64_t m, int64_t k, int64_t n,
                    const double* A, const double* B, double* C, double beta) {
#if defined(LLM_CPP_USE_BLAS)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k, 1.0, A, (int)k, B, (int)n, beta, C, (int)n);
#else
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int64_t p = 0; p < k; ++p) acc += A[i * k + p] * B[p * n + j];
            C[i * n + j] = beta * C[i * n + j] + acc;
        }
#endif
}

// C[m,k] += dC[m,n] * B[k,n]^T  （dA = dC · B^T）
inline void gemm_nt_acc(int64_t m, int64_t k, int64_t n,
                        const double* dC, const double* B, double* C) {
#if defined(LLM_CPP_USE_BLAS)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)k, (int)n, 1.0, dC, (int)n, B, (int)n, 1.0, C, (int)k);
#else
    for (int64_t i = 0; i < m; ++i)
        for (int64_t p = 0; p < k; ++p) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += dC[i * n + j] * B[p * n + j];
            C[i * k + p] += s;
        }
#endif
}

// C[k,n] += A[m,k]^T * dC[m,n]  （dB = A^T · dC）
inline void gemm_tn_acc(int64_t m, int64_t k, int64_t n,
                        const double* A, const double* dC, double* C) {
#if defined(LLM_CPP_USE_BLAS)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)k, (int)n, (int)m, 1.0, A, (int)k, dC, (int)n, 1.0, C, (int)n);
#else
    for (int64_t p = 0; p < k; ++p)
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < m; ++i) s += A[i * k + p] * dC[i * n + j];
            C[p * n + j] += s;
        }
#endif
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("matmul expects [m,k] x [k,n]");
    }
    int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    Tensor out({m, n}, DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    gemm_nn(m, k, n, a.data().data(), b.data().data(), out.data().data(), 0.0);
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, m, k, n]() mutable {
            const double* og = out.grad().data();
            if (a.requires_grad()) {
                gemm_nt_acc(m, k, n, og, b.data().data(), a.grad().data());
            }
            if (b.requires_grad()) {
                gemm_tn_acc(m, k, n, a.data().data(), og, b.grad().data());
            }
        };
    }
    return out;
}

Tensor batch_matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 4 || b.shape().size() != 4) {
        throw std::runtime_error("batch_matmul expects 4D tensors");
    }
    int64_t B = a.shape()[0], H = a.shape()[1], M = a.shape()[2], K = a.shape()[3];
    if (b.shape()[0] != B || b.shape()[1] != H || b.shape()[2] != K) {
        throw std::runtime_error("batch_matmul shape mismatch");
    }
    int64_t N = b.shape()[3];
    Tensor out({B, H, M, N}, DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    {
        const double* ap = a.data().data();
        const double* bp = b.data().data();
        double* op = out.data().data();
        for (int64_t bb = 0; bb < B; ++bb) {
            for (int64_t hh = 0; hh < H; ++hh) {
                gemm_nn(M, K, N,
                        ap + ((bb * H + hh) * M) * K,
                        bp + ((bb * H + hh) * K) * N,
                        op + ((bb * H + hh) * M) * N, 0.0);
            }
        }
    }
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, B, H, M, K, N]() mutable {
            const double* og = out.grad().data();
            if (a.requires_grad()) {
                const double* bp = b.data().data();
                double* ag = a.grad().data();
                for (int64_t bb = 0; bb < B; ++bb) {
                    for (int64_t hh = 0; hh < H; ++hh) {
                        gemm_nt_acc(M, K, N,
                                    og + ((bb * H + hh) * M) * N,
                                    bp + ((bb * H + hh) * K) * N,
                                    ag + ((bb * H + hh) * M) * K);
                    }
                }
            }
            if (b.requires_grad()) {
                const double* ap = a.data().data();
                double* bg = b.grad().data();
                for (int64_t bb = 0; bb < B; ++bb) {
                    for (int64_t hh = 0; hh < H; ++hh) {
                        gemm_tn_acc(M, K, N,
                                    ap + ((bb * H + hh) * M) * K,
                                    og + ((bb * H + hh) * M) * N,
                                    bg + ((bb * H + hh) * K) * N);
                    }
                }
            }
        };
    }
    return out;
}

} // namespace cpu
} // namespace llm
