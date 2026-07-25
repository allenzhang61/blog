#include "llm/ops.hpp"

#include "../kernels/cpu/cpu_ops.hpp"
#include "../kernels/cuda/cuda_ops.hpp"
#include "../kernels/metal/metal_ops.hpp"

namespace llm {
namespace ops {

// ops 层只负责按设备把算子分发到具体后端（cpu:: / cuda:: / metal::），本身不做计算。

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
    if (a.device().type == DeviceType::Metal) {
        return metal::mul_scalar(a, scalar);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::mul_scalar(a, scalar);
    }
    return cpu::mul_scalar(a, scalar);
}

// 分发逐元素幂运算算子。
Tensor pow(const Tensor& a, double exponent) {
    if (a.device().type == DeviceType::Metal) {
        return metal::pow(a, exponent);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::pow(a, exponent);
    }
    return cpu::pow(a, exponent);
}

// 分发张量求和归约算子。
Tensor sum(const Tensor& a) {
    if (a.device().type == DeviceType::Metal) {
        return metal::sum(a);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::sum(a);
    }
    return cpu::sum(a);
}

// 分发张量均值归约算子。
Tensor mean(const Tensor& a) {
    if (a.device().type == DeviceType::Metal) {
        return metal::mean(a);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::mean(a);
    }
    return cpu::mean(a);
}

// 分发张量最大值归约算子。
Tensor max(const Tensor& a) {
    if (a.device().type == DeviceType::Metal) {
        return metal::max(a);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::max(a);
    }
    return cpu::max(a);
}

// 分发张量 reshape 算子。
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (a.device().type == DeviceType::Metal) {
        return metal::reshape(a, new_shape);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::reshape(a, new_shape);
    }
    return cpu::reshape(a, new_shape);
}

// 分发张量维度交换算子。
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
    if (a.device().type == DeviceType::Metal) {
        return metal::transpose(a, dim0, dim1);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::transpose(a, dim0, dim1);
    }
    return cpu::transpose(a, dim0, dim1);
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
    if (scores.device().type == DeviceType::Metal) {
        return metal::causal_mask(scores, sequence_length, mask_value);
    }
    if (scores.device().type == DeviceType::CUDA) {
        return cuda::causal_mask(scores, sequence_length, mask_value);
    }
    return cpu::causal_mask(scores, sequence_length, mask_value);
}

// 分发 softmax 算子。
Tensor softmax(const Tensor& a, int64_t dim) {
    if (a.device().type == DeviceType::Metal) {
        return metal::softmax(a, dim);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::softmax(a, dim);
    }
    return cpu::softmax(a, dim);
}

// 分发 log_softmax 算子。
Tensor log_softmax(const Tensor& a, int64_t dim) {
    if (a.device().type == DeviceType::Metal) {
        return metal::log_softmax(a, dim);
    }
    if (a.device().type == DeviceType::CUDA) {
        return cuda::log_softmax(a, dim);
    }
    return cpu::log_softmax(a, dim);
}

// 分发交叉熵损失算子。
Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    if (logits.device().type != targets.device().type) {
        throw std::runtime_error("cross_entropy expects logits and targets on the same device");
    }
    if (logits.device().type == DeviceType::Metal) {
        return metal::cross_entropy(logits, targets);
    }
    if (logits.device().type == DeviceType::CUDA) {
        return cuda::cross_entropy(logits, targets);
    }
    return cpu::cross_entropy(logits, targets);
}

// 分发词表 embedding 查表算子。
Tensor embedding(const Tensor& ids, const Tensor& weight) {
    if (ids.device().type != weight.device().type) {
        throw std::runtime_error("embedding expects ids and weight on the same device");
    }
    if (ids.device().type == DeviceType::Metal) {
        return metal::embedding(ids, weight);
    }
    if (ids.device().type == DeviceType::CUDA) {
        return cuda::embedding(ids, weight);
    }
    return cpu::embedding(ids, weight);
}

// 分发 layer normalization 算子。
Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    if (x.device().type != scale.device().type || x.device().type != shift.device().type) {
        throw std::runtime_error("layernorm expects tensors on the same device");
    }
    if (x.device().type == DeviceType::Metal) {
        return metal::layernorm(x, scale, shift, eps);
    }
    if (x.device().type == DeviceType::CUDA) {
        return cuda::layernorm(x, scale, shift, eps);
    }
    return cpu::layernorm(x, scale, shift, eps);
}

// 分发 GELU 激活函数算子。
Tensor gelu(const Tensor& x) {
    if (x.device().type == DeviceType::Metal) {
        return metal::gelu(x);
    }
    if (x.device().type == DeviceType::CUDA) {
        return cuda::gelu(x);
    }
    return cpu::gelu(x);
}

} // namespace ops
} // namespace llm
