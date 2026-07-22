#include "llm/backend/Backend.hpp"
#include "../../kernels/cuda/cuda_ops.hpp"

// 这个文件是「未启用 CUDA 编译」时使用的 stub。
// CMake 在 LLM_CPP_ENABLE_CUDA=ON 且找到 CUDA Toolkit 时，会改用：
// - src/backend/cuda/CUDABackend.cpp
// - src/kernels/cuda/cuda_kernels.cu
//
// 真实 GPU kernel 源码在：
// - src/kernels/cuda/cuda_kernels.cu
//
// 因此下面每个函数都只抛错，但注释会标出启用 CUDA 后的真实实现位置。

namespace llm {

bool cuda_backend_available() {
    // CUDA 开启时对应：
    // src/backend/cuda/CUDABackend.cpp::cuda_backend_available()
    // -> src/kernels/cuda/cuda_kernels.cu::llm::cuda::available()
    return false;
}

std::string cuda_backend_status() {
    // CUDA 开启时对应：
    // src/backend/cuda/CUDABackend.cpp::cuda_backend_status()
    // -> src/kernels/cuda/cuda_kernels.cu::llm::cuda::status()
    return "CUDA backend is unavailable: project was not compiled with LLM_CPP_ENABLE_CUDA=ON or CUDA Toolkit was not found";
}

} // namespace llm

namespace llm::cuda {

bool available() {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::available()
    return false;
}

std::string status() {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::status()
    return cuda_backend_status();
}

Tensor add(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::add()
    // -> CudaRuntime::elementwise2("add", ...) -> add_kernel()
    throw std::runtime_error(status());
}

Tensor sub(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::sub()
    // -> CudaRuntime::unary("neg", ...) -> neg_kernel()，然后复用 llm::cuda::add()。
    throw std::runtime_error(status());
}

Tensor mul(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::mul() -> mul_kernel()
    throw std::runtime_error(status());
}

Tensor div(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::div() -> div_kernel()
    throw std::runtime_error(status());
}

Tensor mul_scalar(const Tensor&, double) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::mul_scalar() -> mul_scalar_kernel()
    throw std::runtime_error(status());
}

Tensor pow(const Tensor&, double) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::pow() -> pow_kernel()
    throw std::runtime_error(status());
}

Tensor sum(const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::sum() -> sum_kernel()
    throw std::runtime_error(status());
}

Tensor mean(const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::mean() -> 先调用 sum_kernel()，均值除法在 host 侧完成。
    throw std::runtime_error(status());
}

Tensor max(const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::max() -> max_kernel()
    throw std::runtime_error(status());
}

Tensor reshape(const Tensor&, const std::vector<int64_t>&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::reshape() -> copy_kernel()
    throw std::runtime_error(status());
}

Tensor transpose(const Tensor&, int64_t, int64_t) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::transpose() -> gather_kernel()
    throw std::runtime_error(status());
}

Tensor matmul(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::matmul() -> matmul_kernel()
    throw std::runtime_error(status());
}

Tensor batch_matmul(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::batch_matmul() -> batch_matmul_kernel()
    throw std::runtime_error(status());
}

Tensor causal_mask(const Tensor&, int64_t, double) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::causal_mask() -> causal_mask_kernel()
    throw std::runtime_error(status());
}

Tensor softmax(const Tensor&, int64_t) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::softmax() -> softmax_kernel()
    throw std::runtime_error(status());
}

Tensor log_softmax(const Tensor&, int64_t) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::log_softmax() -> softmax_kernel() -> log_kernel()
    throw std::runtime_error(status());
}

Tensor cross_entropy(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::cross_entropy() -> cross_entropy_row_loss_kernel()
    throw std::runtime_error(status());
}

Tensor embedding(const Tensor&, const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::embedding() -> embedding_kernel()
    throw std::runtime_error(status());
}

Tensor layernorm(const Tensor&, const Tensor&, const Tensor&, double) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::layernorm() -> layernorm_kernel()
    throw std::runtime_error(status());
}

Tensor gelu(const Tensor&) {
    // CUDA 开启时对应：
    // src/kernels/cuda/cuda_kernels.cu::llm::cuda::gelu() -> gelu_kernel()
    throw std::runtime_error(status());
}

} // namespace llm::cuda
