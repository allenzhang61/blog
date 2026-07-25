#include "cuda_runtime.hpp"
#include "cuda_device_kernels.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace {

// 统一的 CUDA 调用错误检查。
void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

constexpr int kThreadsPerBlock = 256;

int blocks_for(int64_t count) {
    return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

} // namespace

namespace llm::cuda::detail {

DeviceBuffer::DeviceBuffer(size_t n) : count(n) {
    if (n > 0) {
        cuda_check(cudaMalloc(&ptr, n * sizeof(float)), "cudaMalloc");
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), count(other.count) {
    other.ptr = nullptr;
    other.count = 0;
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

namespace {

DeviceBuffer upload(const std::vector<float>& host) {
    DeviceBuffer buf(host.size());
    if (!host.empty()) {
        cuda_check(cudaMemcpy(buf.ptr, host.data(), host.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
    }
    return buf;
}

std::vector<float> download(const DeviceBuffer& buf) {
    std::vector<float> host(buf.count);
    if (buf.count > 0) {
        cuda_check(cudaMemcpy(host.data(), buf.ptr, buf.count * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    }
    return host;
}

} // namespace

CudaRuntime& CudaRuntime::instance() {
    static CudaRuntime runtime;
    return runtime;
}

bool CudaRuntime::available() const {
    return available_;
}

std::string CudaRuntime::status() const {
    return status_;
}

void CudaRuntime::require() const {
    if (!available_) {
        throw std::runtime_error(status_);
    }
}

namespace {

void release_tensor_storage(TensorCudaStorage& storage) {
    if (storage.data != nullptr) {
        cudaFree(storage.data);
        storage.data = nullptr;
    }
    if (storage.grad != nullptr) {
        cudaFree(storage.grad);
        storage.grad = nullptr;
    }
    storage.data_count = 0;
    storage.grad_count = 0;
}

std::vector<float> double_to_float(const std::vector<double>& values) {
    return std::vector<float>(values.begin(), values.end());
}

void copy_from_host_to_float_buffer(void* ptr, const std::vector<double>& host, const char* what) {
    if (host.empty()) {
        return;
    }
    std::vector<float> values = double_to_float(host);
    cuda_check(cudaMemcpy(ptr, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), what);
}

void copy_from_float_buffer_to_host(void* ptr, size_t count, std::vector<double>& host, const char* what) {
    host.assign(count, 0.0);
    if (count == 0) {
        return;
    }
    std::vector<float> values(count);
    cuda_check(cudaMemcpy(values.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost), what);
    host.assign(values.begin(), values.end());
}

void ensure_float_buffer(void*& ptr, size_t& current_count, size_t requested_count, const char* what) {
    if (current_count >= requested_count) {
        return;
    }
    if (ptr != nullptr) {
        cuda_check(cudaFree(ptr), what);
        ptr = nullptr;
        current_count = 0;
    }
    if (requested_count > 0) {
        cuda_check(cudaMalloc(&ptr, requested_count * sizeof(float)), what);
    }
    current_count = requested_count;
}

void tensor_copy_data_from_host(TensorCudaStorage& storage, const std::vector<double>& host) {
    ensure_float_buffer(storage.data, storage.data_count, host.size(), "cuda tensor data buffer");
    copy_from_host_to_float_buffer(storage.data, host, "cuda tensor data H2D");
}

void tensor_copy_data_to_host(TensorCudaStorage& storage, std::vector<double>& host) {
    copy_from_float_buffer_to_host(storage.data, storage.data_count, host, "cuda tensor data D2H");
}

void tensor_copy_grad_from_host(TensorCudaStorage& storage, const std::vector<double>& host) {
    ensure_float_buffer(storage.grad, storage.grad_count, host.size(), "cuda tensor grad buffer");
    copy_from_host_to_float_buffer(storage.grad, host, "cuda tensor grad H2D");
}

void tensor_copy_grad_to_host(TensorCudaStorage& storage, std::vector<double>& host) {
    copy_from_float_buffer_to_host(storage.grad, storage.grad_count, host, "cuda tensor grad D2H");
}

} // namespace

std::shared_ptr<TensorCudaStorage> CudaRuntime::create_tensor_storage() {
    require();
    auto storage = std::make_shared<TensorCudaStorage>();
    storage->release = release_tensor_storage;
    storage->copy_data_from_host = tensor_copy_data_from_host;
    storage->copy_data_to_host = tensor_copy_data_to_host;
    storage->copy_grad_from_host = tensor_copy_grad_from_host;
    storage->copy_grad_to_host = tensor_copy_grad_to_host;
    storage->fill_grad = [](TensorCudaStorage& s, size_t count, float value) {
        CudaRuntime::instance().fill_grad_buffer(s, count, value);
    };
    return storage;
}

void CudaRuntime::ensure_data_buffer(TensorCudaStorage& storage, size_t count) {
    require();
    ensure_float_buffer(storage.data, storage.data_count, count, "cuda tensor data buffer");
}

void CudaRuntime::ensure_grad_buffer(TensorCudaStorage& storage, size_t count) {
    require();
    ensure_float_buffer(storage.grad, storage.grad_count, count, "cuda tensor grad buffer");
}

void CudaRuntime::copy_data_from_host(TensorCudaStorage& storage, const std::vector<double>& host) {
    require();
    tensor_copy_data_from_host(storage, host);
}

void CudaRuntime::copy_data_to_host(TensorCudaStorage& storage, std::vector<double>& host) {
    require();
    tensor_copy_data_to_host(storage, host);
}

void CudaRuntime::copy_grad_from_host(TensorCudaStorage& storage, const std::vector<double>& host) {
    require();
    tensor_copy_grad_from_host(storage, host);
}

void CudaRuntime::copy_grad_to_host(TensorCudaStorage& storage, std::vector<double>& host) {
    require();
    tensor_copy_grad_to_host(storage, host);
}

void CudaRuntime::fill_data_buffer(TensorCudaStorage& storage, size_t count, float value) {
    require();
    ensure_data_buffer(storage, count);
    fill_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(storage.data), value, static_cast<long long>(count));
    sync();
}

void CudaRuntime::fill_grad_buffer(TensorCudaStorage& storage, size_t count, float value) {
    require();
    ensure_grad_buffer(storage, count);
    fill_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(storage.grad), value, static_cast<long long>(count));
    sync();
}

void CudaRuntime::set_data_scalar(TensorCudaStorage& storage, float value) {
    fill_data_buffer(storage, 1, value);
}

void CudaRuntime::set_grad_scalar(TensorCudaStorage& storage, float value) {
    fill_grad_buffer(storage, 1, value);
}

void CudaRuntime::elementwise2_buffer(const char* op, TensorCudaStorage& out,
                                      const TensorCudaStorage& a, const TensorCudaStorage& b,
                                      unsigned int b_size, size_t count) {
    require();
    ensure_data_buffer(out, count);
    int blocks = blocks_for(static_cast<int64_t>(count));
    if (std::string(op) == "add") {
        add_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                 static_cast<const float*>(b.data),
                                                 static_cast<float*>(out.data), b_size,
                                                 static_cast<long long>(count));
    } else if (std::string(op) == "mul") {
        mul_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                 static_cast<const float*>(b.data),
                                                 static_cast<float*>(out.data),
                                                 static_cast<long long>(count));
    } else {
        div_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                 static_cast<const float*>(b.data),
                                                 static_cast<float*>(out.data),
                                                 static_cast<long long>(count));
    }
    sync();
}

void CudaRuntime::mul_scalar_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                                    float scalar, size_t count) {
    require();
    ensure_data_buffer(out, count);
    mul_scalar_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<const float*>(a.data), static_cast<float*>(out.data), scalar,
        static_cast<long long>(count));
    sync();
}

void CudaRuntime::unary_buffer(const char* op, TensorCudaStorage& out, const TensorCudaStorage& a,
                               float scalar, size_t count) {
    require();
    ensure_data_buffer(out, count);
    int blocks = blocks_for(static_cast<int64_t>(count));
    std::string name(op);
    if (name == "neg") {
        neg_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                 static_cast<float*>(out.data),
                                                 static_cast<long long>(count));
    } else if (name == "pow") {
        pow_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                 static_cast<float*>(out.data), scalar,
                                                 static_cast<long long>(count));
    } else if (name == "copy") {
        copy_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                  static_cast<float*>(out.data),
                                                  static_cast<long long>(count));
    } else if (name == "log") {
        log_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                 static_cast<float*>(out.data),
                                                 static_cast<long long>(count));
    } else {
        gelu_kernel<<<blocks, kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                  static_cast<float*>(out.data),
                                                  static_cast<long long>(count));
    }
    sync();
}

void CudaRuntime::gather_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                                const std::vector<unsigned int>& index) {
    require();
    int64_t count = static_cast<int64_t>(index.size());
    unsigned int* d_index = nullptr;
    cuda_check(cudaMalloc(&d_index, index.size() * sizeof(unsigned int)), "cudaMalloc index");
    cuda_check(cudaMemcpy(d_index, index.data(), index.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy index");
    ensure_data_buffer(out, index.size());
    gather_kernel<<<blocks_for(count), kThreadsPerBlock>>>(static_cast<const float*>(a.data), d_index,
                                                           static_cast<float*>(out.data), count);
    sync();
    cudaFree(d_index);
}

void CudaRuntime::scale_data_buffer(TensorCudaStorage& storage, size_t count, float scalar) {
    require();
    mul_scalar_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<const float*>(storage.data), static_cast<float*>(storage.data), scalar,
        static_cast<long long>(count));
    sync();
}

void CudaRuntime::reduce_buffer(const char* op, TensorCudaStorage& out,
                                const TensorCudaStorage& a, size_t count) {
    require();
    ensure_data_buffer(out, 1);
    if (std::string(op) == "sum") {
        sum_kernel<<<1, 1>>>(static_cast<const float*>(a.data), static_cast<float*>(out.data),
                             static_cast<unsigned int>(count));
    } else {
        max_kernel<<<1, 1>>>(static_cast<const float*>(a.data), static_cast<float*>(out.data),
                             static_cast<unsigned int>(count));
    }
    sync();
}

void CudaRuntime::matmul_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                                const TensorCudaStorage& b, unsigned int m, unsigned int k,
                                unsigned int n) {
    require();
    ensure_data_buffer(out, static_cast<size_t>(m) * n);
    dim3 threads(16, 16, 1);
    dim3 grid((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y, 1);
    matmul_kernel<<<grid, threads>>>(static_cast<const float*>(a.data),
                                     static_cast<const float*>(b.data),
                                     static_cast<float*>(out.data), m, k, n);
    sync();
}

void CudaRuntime::batch_matmul_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                                      const TensorCudaStorage& b, unsigned int batches,
                                      unsigned int heads, unsigned int m, unsigned int k,
                                      unsigned int n) {
    require();
    int64_t total = static_cast<int64_t>(batches) * heads * m * n;
    ensure_data_buffer(out, static_cast<size_t>(total));
    batch_matmul_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
        static_cast<const float*>(a.data), static_cast<const float*>(b.data),
        static_cast<float*>(out.data), batches, heads, m, k, n);
    sync();
}

void CudaRuntime::softmax_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                                 unsigned int rows, unsigned int width) {
    require();
    ensure_data_buffer(out, static_cast<size_t>(rows) * width);
    softmax_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                           static_cast<float*>(out.data), rows, width);
    sync();
}

void CudaRuntime::log_softmax_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                                     unsigned int rows, unsigned int width) {
    require();
    ensure_data_buffer(out, static_cast<size_t>(rows) * width);
    log_softmax_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(static_cast<const float*>(a.data),
                                                               static_cast<float*>(out.data), rows, width);
    sync();
}

void CudaRuntime::layernorm_buffer(TensorCudaStorage& out, const TensorCudaStorage& x,
                                   const TensorCudaStorage& scale, const TensorCudaStorage& shift,
                                   unsigned int rows, unsigned int width, float eps) {
    require();
    ensure_data_buffer(out, static_cast<size_t>(rows) * width);
    layernorm_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(
        static_cast<const float*>(x.data), static_cast<const float*>(scale.data),
        static_cast<const float*>(shift.data), static_cast<float*>(out.data), rows, width, eps);
    sync();
}

void CudaRuntime::embedding_buffer(TensorCudaStorage& out, const TensorCudaStorage& ids,
                                   const TensorCudaStorage& weight, unsigned int count,
                                   unsigned int dim) {
    require();
    int64_t total = static_cast<int64_t>(count) * dim;
    ensure_data_buffer(out, static_cast<size_t>(total));
    embedding_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
        static_cast<const float*>(ids.data), static_cast<const float*>(weight.data),
        static_cast<float*>(out.data), count, dim);
    sync();
}

void CudaRuntime::cross_entropy_loss_buffer(TensorCudaStorage& out, const TensorCudaStorage& logits,
                                            const TensorCudaStorage& targets, unsigned int rows,
                                            unsigned int vocab) {
    require();
    ensure_data_buffer(out, 1);
    cross_entropy_loss_kernel<<<1, 1>>>(static_cast<const float*>(logits.data),
                                        static_cast<const float*>(targets.data),
                                        static_cast<float*>(out.data), rows, vocab);
    sync();
}

void CudaRuntime::add_grad(TensorCudaStorage& target, const TensorCudaStorage& out_grad,
                           unsigned int target_size, size_t count, float scale) {
    require();
    add_grad_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(target.grad), static_cast<const float*>(out_grad.grad), target_size,
        static_cast<long long>(count), scale);
    sync();
}

void CudaRuntime::elementwise_grad(const char* op, TensorCudaStorage* a_grad, TensorCudaStorage* b_grad,
                                   const TensorCudaStorage& a, const TensorCudaStorage& b,
                                   const TensorCudaStorage& out_grad, size_t count) {
    require();
    int blocks = blocks_for(static_cast<int64_t>(count));
    if (std::string(op) == "mul") {
        mul_grad_kernel<<<blocks, kThreadsPerBlock>>>(
            a_grad ? static_cast<float*>(a_grad->grad) : nullptr,
            b_grad ? static_cast<float*>(b_grad->grad) : nullptr,
            static_cast<const float*>(a.data), static_cast<const float*>(b.data),
            static_cast<const float*>(out_grad.grad), static_cast<long long>(count));
    } else {
        div_grad_kernel<<<blocks, kThreadsPerBlock>>>(
            a_grad ? static_cast<float*>(a_grad->grad) : nullptr,
            b_grad ? static_cast<float*>(b_grad->grad) : nullptr,
            static_cast<const float*>(a.data), static_cast<const float*>(b.data),
            static_cast<const float*>(out_grad.grad), static_cast<long long>(count));
    }
    sync();
}

void CudaRuntime::mul_scalar_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out_grad,
                                  float scalar, size_t count) {
    require();
    mul_scalar_grad_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(a_grad.grad), static_cast<const float*>(out_grad.grad), scalar,
        static_cast<long long>(count));
    sync();
}

void CudaRuntime::pow_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& a,
                           const TensorCudaStorage& out_grad, float exponent, size_t count) {
    require();
    pow_grad_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(a_grad.grad), static_cast<const float*>(a.data),
        static_cast<const float*>(out_grad.grad), exponent, static_cast<long long>(count));
    sync();
}

void CudaRuntime::reduce_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out_grad,
                              size_t count, float scale) {
    require();
    reduce_grad_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(a_grad.grad), static_cast<const float*>(out_grad.grad),
        static_cast<long long>(count), scale);
    sync();
}

void CudaRuntime::scatter_add_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out_grad,
                                   const std::vector<unsigned int>& index) {
    require();
    unsigned int* d_index = nullptr;
    cuda_check(cudaMalloc(&d_index, index.size() * sizeof(unsigned int)), "cudaMalloc index");
    cuda_check(cudaMemcpy(d_index, index.data(), index.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy index");
    scatter_add_grad_kernel<<<blocks_for(static_cast<int64_t>(index.size())), kThreadsPerBlock>>>(
        static_cast<float*>(a_grad.grad), static_cast<const float*>(out_grad.grad), d_index,
        static_cast<long long>(index.size()));
    sync();
    cudaFree(d_index);
}

void CudaRuntime::matmul_grad(TensorCudaStorage* a_grad, TensorCudaStorage* b_grad,
                              const TensorCudaStorage& a, const TensorCudaStorage& b,
                              const TensorCudaStorage& out_grad, unsigned int m, unsigned int k,
                              unsigned int n) {
    require();
    dim3 threads(16, 16, 1);
    if (a_grad != nullptr) {
        dim3 grid((k + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y, 1);
        matmul_grad_a_kernel<<<grid, threads>>>(static_cast<float*>(a_grad->grad),
                                                static_cast<const float*>(b.data),
                                                static_cast<const float*>(out_grad.grad), m, k, n);
    }
    if (b_grad != nullptr) {
        dim3 grid((n + threads.x - 1) / threads.x, (k + threads.y - 1) / threads.y, 1);
        matmul_grad_b_kernel<<<grid, threads>>>(static_cast<float*>(b_grad->grad),
                                                static_cast<const float*>(a.data),
                                                static_cast<const float*>(out_grad.grad), m, k, n);
    }
    sync();
}

void CudaRuntime::batch_matmul_grad(TensorCudaStorage* a_grad, TensorCudaStorage* b_grad,
                                    const TensorCudaStorage& a, const TensorCudaStorage& b,
                                    const TensorCudaStorage& out_grad, unsigned int batches,
                                    unsigned int heads, unsigned int m, unsigned int k,
                                    unsigned int n) {
    require();
    if (a_grad != nullptr) {
        int64_t total = static_cast<int64_t>(batches) * heads * m * k;
        batch_matmul_grad_a_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
            static_cast<float*>(a_grad->grad), static_cast<const float*>(b.data),
            static_cast<const float*>(out_grad.grad), batches, heads, m, k, n);
    }
    if (b_grad != nullptr) {
        int64_t total = static_cast<int64_t>(batches) * heads * k * n;
        batch_matmul_grad_b_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
            static_cast<float*>(b_grad->grad), static_cast<const float*>(a.data),
            static_cast<const float*>(out_grad.grad), batches, heads, m, k, n);
    }
    sync();
}

void CudaRuntime::softmax_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out,
                               const TensorCudaStorage& out_grad, unsigned int rows,
                               unsigned int width) {
    require();
    softmax_grad_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(
        static_cast<float*>(a_grad.grad), static_cast<const float*>(out.data),
        static_cast<const float*>(out_grad.grad), rows, width);
    sync();
}

void CudaRuntime::cross_entropy_grad(TensorCudaStorage& logits_grad, const TensorCudaStorage& logits,
                                     const TensorCudaStorage& targets, const TensorCudaStorage& out_grad,
                                     unsigned int rows, unsigned int vocab) {
    require();
    int64_t total = static_cast<int64_t>(rows) * vocab;
    cross_entropy_grad_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
        static_cast<float*>(logits_grad.grad), static_cast<const float*>(logits.data),
        static_cast<const float*>(targets.data), static_cast<const float*>(out_grad.grad), rows, vocab);
    sync();
}

void CudaRuntime::embedding_grad(TensorCudaStorage& weight_grad, const TensorCudaStorage& ids,
                                 const TensorCudaStorage& out_grad, unsigned int count,
                                 unsigned int dim) {
    require();
    int64_t total = static_cast<int64_t>(count) * dim;
    embedding_grad_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
        static_cast<float*>(weight_grad.grad), static_cast<const float*>(ids.data),
        static_cast<const float*>(out_grad.grad), count, dim);
    sync();
}

void CudaRuntime::layernorm_grad(TensorCudaStorage* x_grad, TensorCudaStorage* scale_grad,
                                 TensorCudaStorage* shift_grad, const TensorCudaStorage& x,
                                 const TensorCudaStorage& scale, const TensorCudaStorage& out_grad,
                                 unsigned int rows, unsigned int width, float eps) {
    require();
    int64_t total = static_cast<int64_t>(rows) * width;
    if (x_grad != nullptr) {
        layernorm_grad_x_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
            static_cast<float*>(x_grad->grad), static_cast<const float*>(x.data),
            static_cast<const float*>(scale.data), static_cast<const float*>(out_grad.grad),
            rows, width, eps);
    }
    if (scale_grad != nullptr || shift_grad != nullptr) {
        layernorm_grad_scale_shift_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
            scale_grad ? static_cast<float*>(scale_grad->grad) : nullptr,
            shift_grad ? static_cast<float*>(shift_grad->grad) : nullptr,
            static_cast<const float*>(x.data), static_cast<const float*>(out_grad.grad),
            rows, width, eps);
    }
    sync();
}

void CudaRuntime::gelu_grad(TensorCudaStorage& x_grad, const TensorCudaStorage& x,
                            const TensorCudaStorage& out_grad, size_t count) {
    require();
    gelu_grad_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(x_grad.grad), static_cast<const float*>(x.data),
        static_cast<const float*>(out_grad.grad), static_cast<long long>(count));
    sync();
}

void CudaRuntime::adamw_update(TensorCudaStorage& param, TensorCudaStorage& grad,
                               TensorCudaStorage& m, TensorCudaStorage& v, size_t count,
                               float lr, float weight_decay, float beta1, float beta2,
                               float eps, float bias_correction1, float bias_correction2) {
    require();
    ensure_data_buffer(m, count);
    ensure_data_buffer(v, count);
    adamw_update_kernel<<<blocks_for(static_cast<int64_t>(count)), kThreadsPerBlock>>>(
        static_cast<float*>(param.data), static_cast<const float*>(grad.grad),
        static_cast<float*>(m.data), static_cast<float*>(v.data), static_cast<long long>(count),
        lr, weight_decay, beta1, beta2, eps, bias_correction1, bias_correction2);
    sync();
}

void CudaRuntime::causal_mask_buffer(TensorCudaStorage& out, const TensorCudaStorage& scores,
                                     unsigned int batches, unsigned int heads,
                                     unsigned int sequence_length, float mask_value) {
    require();
    int64_t total = static_cast<int64_t>(batches) * heads * sequence_length * sequence_length;
    ensure_data_buffer(out, static_cast<size_t>(total));
    causal_mask_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
        static_cast<const float*>(scores.data), static_cast<float*>(out.data),
        batches, heads, sequence_length, mask_value);
    sync();
}

void CudaRuntime::causal_mask_grad(TensorCudaStorage& scores_grad, const TensorCudaStorage& out_grad,
                                   unsigned int batches, unsigned int heads,
                                   unsigned int sequence_length) {
    require();
    int64_t total = static_cast<int64_t>(batches) * heads * sequence_length * sequence_length;
    causal_mask_grad_kernel<<<blocks_for(total), kThreadsPerBlock>>>(
        static_cast<float*>(scores_grad.grad), static_cast<const float*>(out_grad.grad),
        batches, heads, sequence_length);
    sync();
}

std::vector<float> CudaRuntime::elementwise2(const char* op, const std::vector<float>& a,
                                             const std::vector<float>& b, unsigned int b_size) {
    require();
    int64_t count = static_cast<int64_t>(a.size());
    DeviceBuffer da = upload(a);
    DeviceBuffer db = upload(b);
    DeviceBuffer dout(a.size());
    int blocks = blocks_for(count);
    if (std::string(op) == "add") {
        add_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, db.ptr, dout.ptr, b_size, count);
    } else if (std::string(op) == "mul") {
        mul_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, db.ptr, dout.ptr, count);
    } else {
        div_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, db.ptr, dout.ptr, count);
    }
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::mul_scalar(const std::vector<float>& a, float scalar) {
    require();
    int64_t count = static_cast<int64_t>(a.size());
    DeviceBuffer da = upload(a);
    DeviceBuffer dout(a.size());
    mul_scalar_kernel<<<blocks_for(count), kThreadsPerBlock>>>(da.ptr, dout.ptr, scalar, count);
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::unary(const char* op, const std::vector<float>& a, float scalar) {
    require();
    int64_t count = static_cast<int64_t>(a.size());
    DeviceBuffer da = upload(a);
    DeviceBuffer dout(a.size());
    int blocks = blocks_for(count);
    std::string name(op);
    if (name == "neg") {
        neg_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, dout.ptr, count);
    } else if (name == "pow") {
        pow_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, dout.ptr, scalar, count);
    } else if (name == "copy") {
        copy_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, dout.ptr, count);
    } else if (name == "log") {
        log_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, dout.ptr, count);
    } else {
        gelu_kernel<<<blocks, kThreadsPerBlock>>>(da.ptr, dout.ptr, count);
    }
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::gather(const std::vector<float>& a,
                                       const std::vector<unsigned int>& index) {
    require();
    int64_t count = static_cast<int64_t>(index.size());
    DeviceBuffer da = upload(a);
    unsigned int* d_index = nullptr;
    cuda_check(cudaMalloc(&d_index, index.size() * sizeof(unsigned int)), "cudaMalloc index");
    cuda_check(cudaMemcpy(d_index, index.data(), index.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy index");
    DeviceBuffer dout(index.size());
    gather_kernel<<<blocks_for(count), kThreadsPerBlock>>>(da.ptr, d_index, dout.ptr, count);
    sync();
    auto out = download(dout);
    cudaFree(d_index);
    return out;
}

std::vector<float> CudaRuntime::matmul(const std::vector<float>& a, const std::vector<float>& b,
                                       unsigned int m, unsigned int k, unsigned int n) {
    require();
    DeviceBuffer da = upload(a);
    DeviceBuffer db = upload(b);
    DeviceBuffer dout(static_cast<size_t>(m) * n);
    dim3 threads(16, 16, 1);
    dim3 grid((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y, 1);
    matmul_kernel<<<grid, threads>>>(da.ptr, db.ptr, dout.ptr, m, k, n);
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::batch_matmul(const std::vector<float>& a,
                                             const std::vector<float>& b,
                                             unsigned int batches, unsigned int heads,
                                             unsigned int m, unsigned int k, unsigned int n) {
    require();
    int64_t total = static_cast<int64_t>(batches) * heads * m * n;
    DeviceBuffer da = upload(a);
    DeviceBuffer db = upload(b);
    DeviceBuffer dout(static_cast<size_t>(total));
    batch_matmul_kernel<<<blocks_for(total), kThreadsPerBlock>>>(da.ptr, db.ptr, dout.ptr,
                                                                 batches, heads, m, k, n);
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::softmax(const std::vector<float>& x, unsigned int rows,
                                        unsigned int width) {
    require();
    DeviceBuffer dx = upload(x);
    DeviceBuffer dout(x.size());
    softmax_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(dx.ptr, dout.ptr, rows, width);
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::layernorm(const std::vector<float>& x,
                                          const std::vector<float>& scale,
                                          const std::vector<float>& shift,
                                          unsigned int rows, unsigned int width, float eps) {
    require();
    DeviceBuffer dx = upload(x);
    DeviceBuffer ds = upload(scale);
    DeviceBuffer dsh = upload(shift);
    DeviceBuffer dout(x.size());
    layernorm_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(dx.ptr, ds.ptr, dsh.ptr, dout.ptr,
                                                             rows, width, eps);
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::embedding(const std::vector<float>& ids,
                                          const std::vector<float>& weight,
                                          unsigned int count, unsigned int dim) {
    require();
    int64_t total = static_cast<int64_t>(count) * dim;
    DeviceBuffer dids = upload(ids);
    DeviceBuffer dw = upload(weight);
    DeviceBuffer dout(static_cast<size_t>(total));
    embedding_kernel<<<blocks_for(total), kThreadsPerBlock>>>(dids.ptr, dw.ptr, dout.ptr, count,
                                                              dim);
    sync();
    return download(dout);
}

std::vector<float> CudaRuntime::cross_entropy_row_losses(const std::vector<float>& logits,
                                                         const std::vector<float>& targets,
                                                         unsigned int rows, unsigned int vocab) {
    require();
    DeviceBuffer dl = upload(logits);
    DeviceBuffer dt = upload(targets);
    DeviceBuffer dout(rows);
    cross_entropy_row_loss_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(dl.ptr, dt.ptr, dout.ptr,
                                                                          rows, vocab);
    sync();
    return download(dout);
}

float CudaRuntime::reduce(const char* op, const std::vector<float>& a) {
    require();
    DeviceBuffer da = upload(a);
    DeviceBuffer dout(1);
    unsigned int count = static_cast<unsigned int>(a.size());
    if (std::string(op) == "sum") {
        sum_kernel<<<1, 1>>>(da.ptr, dout.ptr, count);
    } else {
        max_kernel<<<1, 1>>>(da.ptr, dout.ptr, count);
    }
    sync();
    return download(dout)[0];
}

CudaRuntime::CudaRuntime() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        available_ = false;
        status_ = "CUDA backend is unavailable: no CUDA device was found";
        return;
    }
    available_ = true;
    status_ = "CUDA backend available";
}

void CudaRuntime::sync() {
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

} // namespace llm::cuda::detail
