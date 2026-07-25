#include "llm/backend/Backend.hpp"
#include "../../kernels/metal/metal_ops.hpp"

#include <stdexcept>

// 这个文件是「未启用 Metal 编译」时使用的 stub。
// CMake 在 LLM_CPP_ENABLE_METAL=ON 且平台为 Apple 时，会改用：
// - src/backend/metal/MetalBackend.mm
// - src/kernels/metal/MetalKernels.mm
//
// 真实 GPU kernel 源码在：
// - src/kernels/metal/metal_kernels.metal
//
// 因此下面每个函数都只抛错，但注释会标出启用 Metal 后的真实实现位置。

namespace llm {

bool metal_backend_available() {
    // Metal 开启时对应：
    // src/backend/metal/MetalBackend.mm::metal_backend_available()
    // -> src/kernels/metal/MetalKernels.mm::llm::metal::available()
    return false;
}

std::string metal_backend_status() {
    // Metal 开启时对应：
    // src/backend/metal/MetalBackend.mm::metal_backend_status()
    // -> src/kernels/metal/MetalKernels.mm::llm::metal::status()
    return "Metal backend is unavailable: project was not compiled with LLM_CPP_ENABLE_METAL=ON on an Apple platform";
}

} // namespace llm

namespace llm::metal {

bool available() {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::available()
    return false;
}

std::string status() {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::status()
    return metal_backend_status();
}

Tensor add(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::add()
    // -> MetalRuntime::run1d("add_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::add_kernel()
    throw std::runtime_error(status());
}

Tensor sub(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::sub()
    // -> MetalRuntime::run1d("neg_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::neg_kernel()
    // 然后复用 llm::metal::add()，最终还会调用 add_kernel()。
    throw std::runtime_error(status());
}

Tensor mul(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::mul()
    // -> MetalRuntime::run1d("mul_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::mul_kernel()
    throw std::runtime_error(status());
}

Tensor div(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::div()
    // -> MetalRuntime::run1d("div_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::div_kernel()
    throw std::runtime_error(status());
}

Tensor mul_scalar(const Tensor&, double) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::mul_scalar()
    // -> MetalRuntime::run1d("mul_scalar_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::mul_scalar_kernel()
    throw std::runtime_error(status());
}

Tensor pow(const Tensor&, double) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::pow()
    // -> MetalRuntime::run1d("pow_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::pow_kernel()
    throw std::runtime_error(status());
}

Tensor sum(const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::sum()
    // -> MetalRuntime::reduce("sum_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::sum_kernel()
    throw std::runtime_error(status());
}

Tensor mean(const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::mean()
    // -> 先调用 llm::metal::sum()
    // -> src/kernels/metal/metal_kernels.metal::sum_kernel()
    // 均值除法在 host 侧完成。
    throw std::runtime_error(status());
}

Tensor max(const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::max()
    // -> MetalRuntime::reduce("max_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::max_kernel()
    throw std::runtime_error(status());
}

Tensor reshape(const Tensor&, const std::vector<int64_t>&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::reshape()
    // -> MetalRuntime::run1d("copy_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::copy_kernel()
    throw std::runtime_error(status());
}

Tensor transpose(const Tensor&, int64_t, int64_t) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::transpose()
    // -> MetalRuntime::gather(...)
    // -> src/kernels/metal/metal_kernels.metal::gather_kernel()
    throw std::runtime_error(status());
}

Tensor matmul(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::matmul()
    // -> MetalRuntime::matmul(...)
    // -> src/kernels/metal/metal_kernels.metal::matmul_kernel()
    throw std::runtime_error(status());
}

Tensor batch_matmul(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::batch_matmul()
    // -> MetalRuntime::batch_matmul(...)
    // -> src/kernels/metal/metal_kernels.metal::batch_matmul_kernel()
    throw std::runtime_error(status());
}

Tensor causal_mask(const Tensor&, int64_t, double) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::causal_mask()
    // -> src/kernels/metal/metal_kernels.metal::causal_mask_kernel()
    throw std::runtime_error(status());
}

Tensor softmax(const Tensor&, int64_t) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::softmax()
    // -> MetalRuntime::softmax(...)
    // -> src/kernels/metal/metal_kernels.metal::softmax_kernel()
    throw std::runtime_error(status());
}

Tensor log_softmax(const Tensor&, int64_t) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::log_softmax()
    // -> MetalRuntime::softmax(...)
    // -> src/kernels/metal/metal_kernels.metal::softmax_kernel()
    // -> MetalRuntime::run1d("log_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::log_kernel()
    throw std::runtime_error(status());
}

Tensor cross_entropy(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::cross_entropy()
    // -> MetalRuntime::cross_entropy_row_losses(...)
    // -> src/kernels/metal/metal_kernels.metal::cross_entropy_row_loss_kernel()
    throw std::runtime_error(status());
}

Tensor embedding(const Tensor&, const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::embedding()
    // -> MetalRuntime::embedding(...)
    // -> src/kernels/metal/metal_kernels.metal::embedding_kernel()
    throw std::runtime_error(status());
}

Tensor layernorm(const Tensor&, const Tensor&, const Tensor&, double) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::layernorm()
    // -> MetalRuntime::layernorm(...)
    // -> src/kernels/metal/metal_kernels.metal::layernorm_kernel()
    throw std::runtime_error(status());
}

Tensor gelu(const Tensor&) {
    // Metal 开启时对应：
    // src/kernels/metal/MetalKernels.mm::llm::metal::gelu()
    // -> MetalRuntime::run1d("gelu_kernel", ...)
    // -> src/kernels/metal/metal_kernels.metal::gelu_kernel()
    throw std::runtime_error(status());
}

} // namespace llm::metal
