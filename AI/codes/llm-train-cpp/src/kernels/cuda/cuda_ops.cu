#include "cuda_ops.hpp"
#include "cuda_runtime.hpp"

#include <cmath>
#include <algorithm>
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

llm::TensorStorage& ensure_cuda_storage(const llm::Tensor& t) {
    if (!t.node->cuda_storage) {
        t.node->cuda_storage = llm::cuda::detail::CudaRuntime::instance().create_tensor_storage();
    }
    return *t.node->cuda_storage;
}

void ensure_cuda_data(const llm::Tensor& t) {
    llm::TensorStorage& storage = ensure_cuda_storage(t);
    if (t.node->host_data_dirty || storage.data == nullptr || storage.data_count < static_cast<size_t>(t.numel())) {
        llm::cuda::detail::CudaRuntime::instance().copy_data_from_host(storage, t.node->data);
        t.node->host_data_dirty = false;
        t.node->device_data_dirty = false;
    }
}

void ensure_cuda_grad(const llm::Tensor& t) {
    llm::TensorStorage& storage = ensure_cuda_storage(t);
    if (t.node->grad.empty()) {
        if (storage.grad == nullptr || storage.grad_count < static_cast<size_t>(t.numel())) {
            llm::cuda::detail::CudaRuntime::instance().fill_grad_buffer(storage, static_cast<size_t>(t.numel()), 0.0f);
        }
        t.node->host_grad_dirty = false;
        t.node->device_grad_dirty = true;
        return;
    }
    if (t.node->host_grad_dirty || storage.grad == nullptr || storage.grad_count < static_cast<size_t>(t.numel())) {
        bool host_grad_is_zero = std::all_of(t.node->grad.begin(), t.node->grad.end(),
                                             [](double value) { return value == 0.0; });
        if ((storage.grad == nullptr || storage.grad_count < static_cast<size_t>(t.numel())) && host_grad_is_zero) {
            llm::cuda::detail::CudaRuntime::instance().fill_grad_buffer(storage, static_cast<size_t>(t.numel()), 0.0f);
        } else {
            llm::cuda::detail::CudaRuntime::instance().copy_grad_from_host(storage, t.node->grad);
        }
        t.node->host_grad_dirty = false;
        t.node->device_grad_dirty = true;
    }
}

void mark_cuda_grad_dirty(const llm::Tensor& t) {
    t.node->host_grad_dirty = false;
    t.node->device_grad_dirty = true;
}

llm::Tensor make_cuda_output(const std::vector<int64_t>& shape, llm::Device device, bool requires_grad) {
    llm::Tensor out;
    out.node->shape = shape;
    out.node->dtype = llm::DType::Float32;
    out.node->device = device;
    out.node->requires_grad = requires_grad;
    out.node->cuda_storage = llm::cuda::detail::CudaRuntime::instance().create_tensor_storage();
    // CUDA outputs keep device storage authoritative. Avoid allocating large
    // host mirrors for intermediates such as normal-profile logits [2,256,50257].
    out.node->host_data_dirty = false;
    out.node->device_data_dirty = true;
    out.node->host_grad_dirty = false;
    out.node->device_grad_dirty = false;
    return out;
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
    ensure_cuda_data(a);
    ensure_cuda_data(b);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad() || b.requires_grad());
    CudaRuntime::instance().elementwise2_buffer("add", *out.node->cuda_storage, *a.node->cuda_storage,
                                                *b.node->cuda_storage, static_cast<unsigned int>(b.numel()),
                                                static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, broadcast_batch, broadcast_last]() mutable {
            ensure_cuda_grad(out);
            if (a.requires_grad()) {
                ensure_cuda_grad(a);
                CudaRuntime::instance().add_grad(*a.node->cuda_storage, *out.node->cuda_storage, 0,
                                                  static_cast<size_t>(out.numel()));
                mark_cuda_grad_dirty(a);
            }
            if (b.requires_grad()) {
                ensure_cuda_grad(b);
                CudaRuntime::instance().add_grad(*b.node->cuda_storage, *out.node->cuda_storage,
                                                  (broadcast_batch || broadcast_last)
                                                      ? static_cast<unsigned int>(b.numel())
                                                      : 0,
                                                  static_cast<size_t>(out.numel()));
                mark_cuda_grad_dirty(b);
            }
        };
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("CUDA mul expects same shape");
    }
    ensure_cuda_data(a);
    ensure_cuda_data(b);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad() || b.requires_grad());
    CudaRuntime::instance().elementwise2_buffer("mul", *out.node->cuda_storage, *a.node->cuda_storage,
                                                *b.node->cuda_storage, 0, static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(a);
            ensure_cuda_data(b);
            llm::TensorStorage* a_grad = nullptr;
            llm::TensorStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_cuda_grad(a);
                a_grad = a.node->cuda_storage.get();
            }
            if (b.requires_grad()) {
                ensure_cuda_grad(b);
                b_grad = b.node->cuda_storage.get();
            }
            CudaRuntime::instance().elementwise_grad("mul", a_grad, b_grad, *a.node->cuda_storage,
                                                     *b.node->cuda_storage, *out.node->cuda_storage,
                                                     static_cast<size_t>(out.numel()));
            if (a_grad) {
                mark_cuda_grad_dirty(a);
            }
            if (b_grad) {
                mark_cuda_grad_dirty(b);
            }
        };
    }
    return out;
}

Tensor mul_scalar(const Tensor& a, double scalar) {
    ensure_cuda_data(a);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad());
    CudaRuntime::instance().mul_scalar_buffer(*out.node->cuda_storage, *a.node->cuda_storage,
                                              static_cast<float>(scalar), static_cast<size_t>(a.numel()));
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, scalar]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(a);
            CudaRuntime::instance().mul_scalar_grad(*a.node->cuda_storage, *out.node->cuda_storage,
                                                    static_cast<float>(scalar),
                                                    static_cast<size_t>(out.numel()));
            mark_cuda_grad_dirty(a);
        };
    }
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("CUDA div expects same shape");
    }
    ensure_cuda_data(a);
    ensure_cuda_data(b);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad() || b.requires_grad());
    CudaRuntime::instance().elementwise2_buffer("div", *out.node->cuda_storage, *a.node->cuda_storage,
                                                *b.node->cuda_storage, 0, static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(a);
            ensure_cuda_data(b);
            llm::TensorStorage* a_grad = nullptr;
            llm::TensorStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_cuda_grad(a);
                a_grad = a.node->cuda_storage.get();
            }
            if (b.requires_grad()) {
                ensure_cuda_grad(b);
                b_grad = b.node->cuda_storage.get();
            }
            CudaRuntime::instance().elementwise_grad("div", a_grad, b_grad, *a.node->cuda_storage,
                                                     *b.node->cuda_storage, *out.node->cuda_storage,
                                                     static_cast<size_t>(out.numel()));
            if (a_grad) {
                mark_cuda_grad_dirty(a);
            }
            if (b_grad) {
                mark_cuda_grad_dirty(b);
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
    ensure_cuda_data(a);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad());
    CudaRuntime::instance().unary_buffer("pow", *out.node->cuda_storage, *a.node->cuda_storage,
                                         static_cast<float>(exponent), static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, exponent]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(a);
            ensure_cuda_grad(a);
            CudaRuntime::instance().pow_grad(*a.node->cuda_storage, *a.node->cuda_storage,
                                             *out.node->cuda_storage, static_cast<float>(exponent),
                                             static_cast<size_t>(out.numel()));
            mark_cuda_grad_dirty(a);
        };
    }
    return out;
}

Tensor sum(const Tensor& a) {
    ensure_cuda_data(a);
    Tensor out = make_cuda_output({}, a.device(), a.requires_grad());
    CudaRuntime::instance().reduce_buffer("sum", *out.node->cuda_storage, *a.node->cuda_storage,
                                          static_cast<size_t>(a.numel()));
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(a);
            CudaRuntime::instance().reduce_grad(*a.node->cuda_storage, *out.node->cuda_storage,
                                                static_cast<size_t>(a.numel()), 1.0f);
            mark_cuda_grad_dirty(a);
        };
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor out = sum(a);
    CudaRuntime::instance().scale_data_buffer(*out.node->cuda_storage, 1, 1.0f / static_cast<float>(a.numel()));
    out.node->device_data_dirty = true;
    if (a.requires_grad()) {
        out.node->backward_fn = [a, out]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(a);
            CudaRuntime::instance().reduce_grad(*a.node->cuda_storage, *out.node->cuda_storage,
                                                static_cast<size_t>(a.numel()),
                                                1.0f / static_cast<float>(a.numel()));
            mark_cuda_grad_dirty(a);
        };
    }
    return out;
}

Tensor max(const Tensor& a) {
    ensure_cuda_data(a);
    Tensor out = make_cuda_output({}, a.device(), false);
    CudaRuntime::instance().reduce_buffer("max", *out.node->cuda_storage, *a.node->cuda_storage,
                                          static_cast<size_t>(a.numel()));
    return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (product(new_shape) != a.numel()) {
        throw std::runtime_error("CUDA reshape numel mismatch");
    }
    ensure_cuda_data(a);
    Tensor out = make_cuda_output(new_shape, a.device(), a.requires_grad());
    CudaRuntime::instance().unary_buffer("copy", *out.node->cuda_storage, *a.node->cuda_storage, 0.0f,
                                         static_cast<size_t>(a.numel()));
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(a);
            CudaRuntime::instance().add_grad(*a.node->cuda_storage, *out.node->cuda_storage, 0,
                                              static_cast<size_t>(out.numel()));
            mark_cuda_grad_dirty(a);
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
    ensure_cuda_data(a);
    Tensor out = make_cuda_output(out_shape, a.device(), a.requires_grad());
    CudaRuntime::instance().transpose_buffer(*out.node->cuda_storage, *a.node->cuda_storage,
                                             shape, dim0, dim1);
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, shape, dim0, dim1]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(a);
            CudaRuntime::instance().transpose_add_grad(*a.node->cuda_storage, *out.node->cuda_storage,
                                                       shape, dim0, dim1);
            mark_cuda_grad_dirty(a);
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
    ensure_cuda_data(a);
    ensure_cuda_data(b);
    Tensor out = make_cuda_output({static_cast<int64_t>(m), static_cast<int64_t>(n)}, a.device(),
                                  a.requires_grad() || b.requires_grad());
    CudaRuntime::instance().matmul_buffer(*out.node->cuda_storage, *a.node->cuda_storage, *b.node->cuda_storage,
                                          m, k, n);
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, m, k, n]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(a);
            ensure_cuda_data(b);
            llm::TensorStorage* a_grad = nullptr;
            llm::TensorStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_cuda_grad(a);
                a_grad = a.node->cuda_storage.get();
            }
            if (b.requires_grad()) {
                ensure_cuda_grad(b);
                b_grad = b.node->cuda_storage.get();
            }
            CudaRuntime::instance().matmul_grad(a_grad, b_grad, *a.node->cuda_storage, *b.node->cuda_storage,
                                                *out.node->cuda_storage, m, k, n);
            if (a_grad) {
                mark_cuda_grad_dirty(a);
            }
            if (b_grad) {
                mark_cuda_grad_dirty(b);
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
    ensure_cuda_data(a);
    ensure_cuda_data(b);
    Tensor out = make_cuda_output({B, H, M, N}, a.device(), a.requires_grad() || b.requires_grad());
    CudaRuntime::instance().batch_matmul_buffer(*out.node->cuda_storage, *a.node->cuda_storage,
                                                *b.node->cuda_storage, B, H, M, K, N);
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, B, H, M, K, N]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(a);
            ensure_cuda_data(b);
            llm::TensorStorage* a_grad = nullptr;
            llm::TensorStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_cuda_grad(a);
                a_grad = a.node->cuda_storage.get();
            }
            if (b.requires_grad()) {
                ensure_cuda_grad(b);
                b_grad = b.node->cuda_storage.get();
            }
            CudaRuntime::instance().batch_matmul_grad(a_grad, b_grad, *a.node->cuda_storage,
                                                      *b.node->cuda_storage, *out.node->cuda_storage,
                                                      B, H, M, K, N);
            if (a_grad) {
                mark_cuda_grad_dirty(a);
            }
            if (b_grad) {
                mark_cuda_grad_dirty(b);
            }
        };
    }
    return out;
}

Tensor causal_mask(const Tensor& scores, int64_t sequence_length, double mask_value) {
    if (scores.shape().size() != 4 || scores.shape()[2] != sequence_length || scores.shape()[3] != sequence_length) {
        throw std::runtime_error("CUDA causal_mask expects scores [B,H,T,T]");
    }
    unsigned int B = static_cast<unsigned int>(scores.shape()[0]);
    unsigned int H = static_cast<unsigned int>(scores.shape()[1]);
    unsigned int T = static_cast<unsigned int>(sequence_length);
    ensure_cuda_data(scores);
    Tensor out = make_cuda_output(scores.shape(), scores.device(), scores.requires_grad());
    CudaRuntime::instance().causal_mask_buffer(*out.node->cuda_storage, *scores.node->cuda_storage,
                                               B, H, T, static_cast<float>(mask_value));
    if (scores.requires_grad()) {
        out.node->parents = {scores};
        out.node->backward_fn = [scores, out, B, H, T]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(scores);
            CudaRuntime::instance().causal_mask_grad(*scores.node->cuda_storage, *out.node->cuda_storage, B, H, T);
            mark_cuda_grad_dirty(scores);
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
    ensure_cuda_data(a);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad());
    CudaRuntime::instance().softmax_buffer(*out.node->cuda_storage, *a.node->cuda_storage, rows, width);
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, rows, width]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_grad(a);
            CudaRuntime::instance().softmax_grad(*a.node->cuda_storage, *out.node->cuda_storage,
                                                 *out.node->cuda_storage, rows, width);
            mark_cuda_grad_dirty(a);
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
    ensure_cuda_data(a);
    Tensor out = make_cuda_output(a.shape(), a.device(), a.requires_grad());
    CudaRuntime::instance().log_softmax_buffer(*out.node->cuda_storage, *a.node->cuda_storage, rows, width);
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

    ensure_cuda_data(logits);
    ensure_cuda_data(targets);
    Tensor out = make_cuda_output({}, logits.device(), logits.requires_grad());
    CudaRuntime::instance().cross_entropy_loss_buffer(*out.node->cuda_storage, *logits.node->cuda_storage,
                                                      *targets.node->cuda_storage,
                                                      static_cast<unsigned int>(B * T),
                                                      static_cast<unsigned int>(V));
    if (logits.requires_grad()) {
        out.node->parents = {logits};
        out.node->backward_fn = [logits, targets, out, B, T, V]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(logits);
            ensure_cuda_data(targets);
            ensure_cuda_grad(logits);
            CudaRuntime::instance().cross_entropy_grad(*logits.node->cuda_storage, *logits.node->cuda_storage,
                                                       *targets.node->cuda_storage, *out.node->cuda_storage,
                                                       static_cast<unsigned int>(B * T),
                                                       static_cast<unsigned int>(V));
            mark_cuda_grad_dirty(logits);
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
    ensure_cuda_data(ids);
    ensure_cuda_data(weight);
    Tensor out = make_cuda_output(out_shape, weight.device(), weight.requires_grad());
    CudaRuntime::instance().embedding_buffer(*out.node->cuda_storage, *ids.node->cuda_storage,
                                             *weight.node->cuda_storage, count, dim);
    if (weight.requires_grad()) {
        out.node->parents = {weight};
        out.node->backward_fn = [ids, weight, out, count, dim]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(ids);
            ensure_cuda_grad(weight);
            CudaRuntime::instance().embedding_grad(*weight.node->cuda_storage, *ids.node->cuda_storage,
                                                   *out.node->cuda_storage, count, dim);
            mark_cuda_grad_dirty(weight);
        };
    }
    return out;
}

Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    unsigned int C = static_cast<unsigned int>(x.shape().back());
    unsigned int rows = static_cast<unsigned int>(x.numel() / C);
    ensure_cuda_data(x);
    ensure_cuda_data(scale);
    ensure_cuda_data(shift);
    Tensor out = make_cuda_output(x.shape(), x.device(),
                                  x.requires_grad() || scale.requires_grad() || shift.requires_grad());
    CudaRuntime::instance().layernorm_buffer(*out.node->cuda_storage, *x.node->cuda_storage,
                                             *scale.node->cuda_storage, *shift.node->cuda_storage,
                                             rows, C, static_cast<float>(eps));

    if (out.requires_grad()) {
        out.node->parents = {x, scale, shift};
        out.node->backward_fn = [x, scale, shift, out, C, rows, eps]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(x);
            ensure_cuda_data(scale);
            llm::TensorStorage* x_grad = nullptr;
            llm::TensorStorage* scale_grad = nullptr;
            llm::TensorStorage* shift_grad = nullptr;
            if (scale.requires_grad()) {
                ensure_cuda_grad(scale);
                scale_grad = scale.node->cuda_storage.get();
            }
            if (shift.requires_grad()) {
                ensure_cuda_grad(shift);
                shift_grad = shift.node->cuda_storage.get();
            }
            if (x.requires_grad()) {
                ensure_cuda_grad(x);
                x_grad = x.node->cuda_storage.get();
            }
            CudaRuntime::instance().layernorm_grad(x_grad, scale_grad, shift_grad, *x.node->cuda_storage,
                                                   *scale.node->cuda_storage, *out.node->cuda_storage,
                                                   rows, C, static_cast<float>(eps));
            if (x_grad) {
                mark_cuda_grad_dirty(x);
            }
            if (scale_grad) {
                mark_cuda_grad_dirty(scale);
            }
            if (shift_grad) {
                mark_cuda_grad_dirty(shift);
            }
        };
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    ensure_cuda_data(x);
    Tensor out = make_cuda_output(x.shape(), x.device(), x.requires_grad());
    CudaRuntime::instance().unary_buffer("gelu", *out.node->cuda_storage, *x.node->cuda_storage, 0.0f,
                                         static_cast<size_t>(x.numel()));
    if (x.requires_grad()) {
        out.node->parents = {x};
        out.node->backward_fn = [x, out]() mutable {
            ensure_cuda_grad(out);
            ensure_cuda_data(x);
            ensure_cuda_grad(x);
            CudaRuntime::instance().gelu_grad(*x.node->cuda_storage, *x.node->cuda_storage,
                                              *out.node->cuda_storage, static_cast<size_t>(x.numel()));
            mark_cuda_grad_dirty(x);
        };
    }
    return out;
}

} // namespace llm::cuda
