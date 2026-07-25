#include "llm/ops.hpp"

#include "../kernels/cpu/cpu_ops.hpp"
#include "../kernels/cuda/cuda_ops.hpp"
#include "../kernels/metal/metal_ops.hpp"

#include <stdexcept>

namespace llm {
namespace ops {

// ops 层只负责按设备把算子分发到具体后端（cpu:: / cuda:: / metal::），本身不做计算。

namespace {

bool uses_device(const Tensor& a, const Tensor& b, DeviceType type);
void expect_device_pair(const Tensor& a, const Tensor& b, DeviceType type, const char* op_name);

template <typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_unary(const Tensor& a, MetalFn metal_fn, CudaFn cuda_fn, CpuFn cpu_fn);

template <typename Arg, typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_unary_arg(const Tensor& a, Arg arg, MetalFn metal_fn, CudaFn cuda_fn, CpuFn cpu_fn);

template <typename Arg0, typename Arg1, typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_unary_args(const Tensor& a, Arg0 arg0, Arg1 arg1, MetalFn metal_fn, CudaFn cuda_fn, CpuFn cpu_fn);

template <typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_binary(const char* op_name,
                       const Tensor& a,
                       const Tensor& b,
                       MetalFn metal_fn,
                       CudaFn cuda_fn,
                       CpuFn cpu_fn);

template <typename Arg, typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_ternary_arg(const char* op_name,
                            const Tensor& a,
                            const Tensor& b,
                            const Tensor& c,
                            Arg arg,
                            MetalFn metal_fn,
                            CudaFn cuda_fn,
                            CpuFn cpu_fn);

} // namespace

// 分发逐元素加法算子。
Tensor add(const Tensor& a, const Tensor& b) {
    return dispatch_binary("add", a, b, metal::add, cuda::add, cpu::add);
}

// 分发逐元素减法算子。
Tensor sub(const Tensor& a, const Tensor& b) {
    return dispatch_binary("sub", a, b, metal::sub, cuda::sub, cpu::sub);
}

// 分发逐元素乘法算子。
Tensor mul(const Tensor& a, const Tensor& b) {
    return dispatch_binary("mul", a, b, metal::mul, cuda::mul, cpu::mul);
}

// 分发逐元素除法算子。
Tensor div(const Tensor& a, const Tensor& b) {
    return dispatch_binary("div", a, b, metal::div, cuda::div, cpu::div);
}

// 分发张量与标量相乘算子。
Tensor mul_scalar(const Tensor& a, double scalar) {
    return dispatch_unary_arg(a, scalar, metal::mul_scalar, cuda::mul_scalar, cpu::mul_scalar);
}

// 分发逐元素幂运算算子。
Tensor pow(const Tensor& a, double exponent) {
    return dispatch_unary_arg(a, exponent, metal::pow, cuda::pow, cpu::pow);
}

// 分发张量求和归约算子。
Tensor sum(const Tensor& a) {
    return dispatch_unary(a, metal::sum, cuda::sum, cpu::sum);
}

// 分发张量均值归约算子。
Tensor mean(const Tensor& a) {
    return dispatch_unary(a, metal::mean, cuda::mean, cpu::mean);
}

// 分发张量最大值归约算子。
Tensor max(const Tensor& a) {
    return dispatch_unary(a, metal::max, cuda::max, cpu::max);
}

// 分发张量 reshape 算子。
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    return dispatch_unary_arg<const std::vector<int64_t>&>(a, new_shape, metal::reshape, cuda::reshape, cpu::reshape);
}

// 分发张量维度交换算子。
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
    return dispatch_unary_args(a, dim0, dim1, metal::transpose, cuda::transpose, cpu::transpose);
}

// 分发二维矩阵乘法算子。
Tensor matmul(const Tensor& a, const Tensor& b) {
    return dispatch_binary("matmul", a, b, metal::matmul, cuda::matmul, cpu::matmul);
}

// 分发批量矩阵乘法算子。
Tensor batch_matmul(const Tensor& a, const Tensor& b) {
    return dispatch_binary("batch_matmul", a, b, metal::batch_matmul, cuda::batch_matmul, cpu::batch_matmul);
}

// 分发 causal mask 算子。
Tensor causal_mask(const Tensor& scores, int64_t sequence_length, double mask_value) {
    return dispatch_unary_args(scores, sequence_length, mask_value, metal::causal_mask, cuda::causal_mask, cpu::causal_mask);
}

// 分发 softmax 算子。
Tensor softmax(const Tensor& a, int64_t dim) {
    return dispatch_unary_arg(a, dim, metal::softmax, cuda::softmax, cpu::softmax);
}

// 分发 log_softmax 算子。
Tensor log_softmax(const Tensor& a, int64_t dim) {
    return dispatch_unary_arg(a, dim, metal::log_softmax, cuda::log_softmax, cpu::log_softmax);
}

// 分发交叉熵损失算子。
Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    return dispatch_binary("cross_entropy", logits, targets, metal::cross_entropy, cuda::cross_entropy, cpu::cross_entropy);
}

// 分发词表 embedding 查表算子。
Tensor embedding(const Tensor& ids, const Tensor& weight) {
    return dispatch_binary("embedding", ids, weight, metal::embedding, cuda::embedding, cpu::embedding);
}

// 分发 layer normalization 算子。
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    return dispatch_ternary_arg("layernorm", x, scale, shift, eps, metal::layernorm, cuda::layernorm, cpu::layernorm);
}

// 分发 GELU 激活函数算子。
Tensor gelu(const Tensor& x) {
    return dispatch_unary(x, metal::gelu, cuda::gelu, cpu::gelu);
}

namespace {

bool uses_device(const Tensor& a, const Tensor& b, DeviceType type) {
    return a.device().type == type || b.device().type == type;
}

void expect_device_pair(const Tensor& a, const Tensor& b, DeviceType type, const char* op_name) {
    if (a.device().type != type || b.device().type != type) {
        throw std::runtime_error(std::string(op_name) + " expects tensors on the same device");
    }
}

template <typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_unary(const Tensor& a, MetalFn metal_fn, CudaFn cuda_fn, CpuFn cpu_fn) {
    if (a.device().type == DeviceType::Metal) {
        return metal_fn(a);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda_fn(a);
    }
    return cpu_fn(a);
}

template <typename Arg, typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_unary_arg(const Tensor& a, Arg arg, MetalFn metal_fn, CudaFn cuda_fn, CpuFn cpu_fn) {
    if (a.device().type == DeviceType::Metal) {
        return metal_fn(a, arg);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda_fn(a, arg);
    }
    return cpu_fn(a, arg);
}

template <typename Arg0, typename Arg1, typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_unary_args(const Tensor& a, Arg0 arg0, Arg1 arg1, MetalFn metal_fn, CudaFn cuda_fn, CpuFn cpu_fn) {
    if (a.device().type == DeviceType::Metal) {
        return metal_fn(a, arg0, arg1);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda_fn(a, arg0, arg1);
    }
    return cpu_fn(a, arg0, arg1);
}

template <typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_binary(const char* op_name,
                       const Tensor& a,
                       const Tensor& b,
                       MetalFn metal_fn,
                       CudaFn cuda_fn,
                       CpuFn cpu_fn) {
    if (uses_device(a, b, DeviceType::Metal)) {
        expect_device_pair(a, b, DeviceType::Metal, op_name);
        return metal_fn(a, b);
    }
    if (uses_device(a, b, DeviceType::CUDA)) {
        expect_device_pair(a, b, DeviceType::CUDA, op_name);
        return cuda_fn(a, b);
    }
    return cpu_fn(a, b);
}

template <typename Arg, typename MetalFn, typename CudaFn, typename CpuFn>
Tensor dispatch_ternary_arg(const char* op_name,
                            const Tensor& a,
                            const Tensor& b,
                            const Tensor& c,
                            Arg arg,
                            MetalFn metal_fn,
                            CudaFn cuda_fn,
                            CpuFn cpu_fn) {
    if (a.device().type != b.device().type || a.device().type != c.device().type) {
        throw std::runtime_error(std::string(op_name) + " expects tensors on the same device");
    }
    if (a.device().type == DeviceType::Metal) {
        return metal_fn(a, b, c, arg);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda_fn(a, b, c, arg);
    }
    return cpu_fn(a, b, c, arg);
}

} // namespace

} // namespace ops
} // namespace llm
