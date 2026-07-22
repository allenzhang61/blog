#include "cuda_runtime.hpp"

#include <cuda_runtime.h>

#include <cmath>
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

// 设备内存 RAII 包装，避免手动 cudaFree 泄漏。
struct DeviceBuffer {
    float* ptr{nullptr};
    size_t count{0};

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : count(n) {
        if (n > 0) {
            cuda_check(cudaMalloc(&ptr, n * sizeof(float)), "cudaMalloc");
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

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

// 设备端 kernel（对应 src/kernels/metal/metal_kernels.metal 中的各个 kernel）。

__global__ void add_kernel(const float* a, const float* b, float* out, unsigned int b_size,
                           long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = a[id] + b[(b_size == 1) ? 0 : id % b_size];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = a[id] * b[id];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* out, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = a[id] / b[id];
    }
}

__global__ void mul_scalar_kernel(const float* a, float* out, float scalar, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = a[id] * scalar;
    }
}

__global__ void neg_kernel(const float* a, float* out, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = -a[id];
    }
}

__global__ void pow_kernel(const float* a, float* out, float exponent, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = powf(a[id], exponent);
    }
}

__global__ void copy_kernel(const float* a, float* out, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = a[id];
    }
}

__global__ void log_kernel(const float* a, float* out, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = logf(fmaxf(a[id], 1.0e-12f));
    }
}

__global__ void gather_kernel(const float* a, const unsigned int* index, float* out,
                              long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = a[index[id]];
    }
}

__global__ void gelu_kernel(const float* x, float* out, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        float v = x[id];
        float u = 0.7978845608f * (v + 0.044715f * v * v * v);
        out[id] = 0.5f * v * (1.0f + tanhf(u));
    }
}

__global__ void matmul_kernel(const float* a, const float* b, float* out, unsigned int m,
                              unsigned int k, unsigned int n) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int p = 0; p < k; ++p) {
        acc += a[row * k + p] * b[p * n + col];
    }
    out[row * n + col] = acc;
}

__global__ void batch_matmul_kernel(const float* a, const float* b, float* out,
                                    unsigned int batches, unsigned int heads, unsigned int m,
                                    unsigned int k, unsigned int n) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batches * heads * m * n;
    if (id >= total) {
        return;
    }
    unsigned int col = id % n;
    unsigned int row = (id / n) % m;
    unsigned int head = (id / ((long long)n * m)) % heads;
    unsigned int batch = id / ((long long)n * m * heads);
    float acc = 0.0f;
    for (unsigned int p = 0; p < k; ++p) {
        long long ai = ((long long)(batch * heads + head) * m + row) * k + p;
        long long bi = ((long long)(batch * heads + head) * k + p) * n + col;
        acc += a[ai] * b[bi];
    }
    out[id] = acc;
}

__global__ void softmax_kernel(const float* x, float* out, unsigned int rows, unsigned int width) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned int base = row * width;
    float mx = -INFINITY;
    for (unsigned int c = 0; c < width; ++c) {
        mx = fmaxf(mx, x[base + c]);
    }
    float denom = 0.0f;
    for (unsigned int c = 0; c < width; ++c) {
        denom += expf(x[base + c] - mx);
    }
    for (unsigned int c = 0; c < width; ++c) {
        out[base + c] = expf(x[base + c] - mx) / denom;
    }
}

__global__ void layernorm_kernel(const float* x, const float* scale, const float* shift,
                                 float* out, unsigned int rows, unsigned int width, float eps) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned int base = row * width;
    float mean = 0.0f;
    for (unsigned int c = 0; c < width; ++c) {
        mean += x[base + c];
    }
    mean /= (float)width;
    float var = 0.0f;
    for (unsigned int c = 0; c < width; ++c) {
        float z = x[base + c] - mean;
        var += z * z;
    }
    float inv = rsqrtf(var / (float)width + eps);
    for (unsigned int c = 0; c < width; ++c) {
        float xhat = (x[base + c] - mean) * inv;
        out[base + c] = xhat * scale[c] + shift[c];
    }
}

__global__ void embedding_kernel(const float* ids, const float* weight, float* out,
                                 unsigned int count, unsigned int dim) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)count * dim;
    if (id >= total) {
        return;
    }
    unsigned int token_index = id / dim;
    unsigned int d = id % dim;
    unsigned int token = (unsigned int)(ids[token_index]);
    out[id] = weight[token * dim + d];
}

__global__ void cross_entropy_row_loss_kernel(const float* logits, const float* targets,
                                              float* row_losses, unsigned int rows,
                                              unsigned int vocab) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned int base = row * vocab;
    float mx = -INFINITY;
    for (unsigned int v = 0; v < vocab; ++v) {
        mx = fmaxf(mx, logits[base + v]);
    }
    float denom = 0.0f;
    for (unsigned int v = 0; v < vocab; ++v) {
        denom += expf(logits[base + v] - mx);
    }
    unsigned int target = (unsigned int)(targets[row]);
    float p = expf(logits[base + target] - mx) / denom;
    row_losses[row] = -logf(fmaxf(p, 1.0e-12f));
}

__global__ void sum_kernel(const float* a, float* out, unsigned int count) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int i = 0; i < count; ++i) {
        acc += a[i];
    }
    out[0] = acc;
}

__global__ void max_kernel(const float* a, float* out, unsigned int count) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) {
        return;
    }
    float mx = -INFINITY;
    for (unsigned int i = 0; i < count; ++i) {
        mx = fmaxf(mx, a[i]);
    }
    out[0] = mx;
}

} // namespace

namespace llm::cuda::detail {

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
