#include "cuda_device_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace llm::cuda::detail {

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

__global__ void fill_kernel(float* out, float value, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        out[id] = value;
    }
}

__global__ void log_softmax_kernel(const float* x, float* out, unsigned int rows, unsigned int width) {
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
    float log_denom = logf(fmaxf(denom, 1.0e-12f));
    for (unsigned int c = 0; c < width; ++c) {
        out[base + c] = x[base + c] - mx - log_denom;
    }
}

__global__ void cross_entropy_loss_kernel(const float* logits, const float* targets, float* loss,
                                          unsigned int rows, unsigned int vocab) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int row = 0; row < rows; ++row) {
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
        acc += -logf(fmaxf(p, 1.0e-12f));
    }
    loss[0] = acc / (float)rows;
}

__global__ void cross_entropy_row_loss_parallel_kernel(const float* logits, const float* targets,
                                                       float* row_losses, unsigned int rows,
                                                       unsigned int vocab) {
    unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned int tid = threadIdx.x;
    unsigned int base = row * vocab;
    __shared__ float scratch[256];

    float local_max = -INFINITY;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        local_max = fmaxf(local_max, logits[base + v]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float mx = scratch[0];

    float local_sum = 0.0f;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        local_sum += expf(logits[base + v] - mx);
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float denom = scratch[0];
    if (tid == 0) {
        unsigned int target = (unsigned int)(targets[row]);
        float p = expf(logits[base + target] - mx) / denom;
        row_losses[row] = -logf(fmaxf(p, 1.0e-12f));
    }
}

__global__ void add_grad_kernel(float* target_grad, const float* out_grad, unsigned int target_size,
                                long long count, float scale) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        unsigned int target = (target_size == 0) ? (unsigned int)id : (unsigned int)(id % target_size);
        atomicAdd(&target_grad[target], out_grad[id] * scale);
    }
}

__global__ void mul_grad_kernel(float* a_grad, float* b_grad, const float* a, const float* b,
                                const float* out_grad, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        float g = out_grad[id];
        if (a_grad != nullptr) {
            atomicAdd(&a_grad[id], b[id] * g);
        }
        if (b_grad != nullptr) {
            atomicAdd(&b_grad[id], a[id] * g);
        }
    }
}

__global__ void div_grad_kernel(float* a_grad, float* b_grad, const float* a, const float* b,
                                const float* out_grad, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        float g = out_grad[id];
        if (a_grad != nullptr) {
            atomicAdd(&a_grad[id], g / b[id]);
        }
        if (b_grad != nullptr) {
            atomicAdd(&b_grad[id], -g * a[id] / (b[id] * b[id]));
        }
    }
}

__global__ void mul_scalar_grad_kernel(float* a_grad, const float* out_grad, float scalar,
                                       long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        atomicAdd(&a_grad[id], out_grad[id] * scalar);
    }
}

__global__ void pow_grad_kernel(float* a_grad, const float* a, const float* out_grad,
                                float exponent, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        atomicAdd(&a_grad[id], exponent * powf(a[id], exponent - 1.0f) * out_grad[id]);
    }
}

__global__ void reduce_grad_kernel(float* a_grad, const float* out_grad, long long count, float scale) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        atomicAdd(&a_grad[id], out_grad[0] * scale);
    }
}

__global__ void scatter_add_grad_kernel(float* a_grad, const float* out_grad,
                                        const unsigned int* index, long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        atomicAdd(&a_grad[index[id]], out_grad[id]);
    }
}

__global__ void matmul_grad_a_kernel(float* a_grad, const float* b, const float* out_grad,
                                     unsigned int m, unsigned int k, unsigned int n) {
    unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || p >= k) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int j = 0; j < n; ++j) {
        acc += out_grad[i * n + j] * b[p * n + j];
    }
    atomicAdd(&a_grad[i * k + p], acc);
}

__global__ void matmul_grad_b_kernel(float* b_grad, const float* a, const float* out_grad,
                                     unsigned int m, unsigned int k, unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= k || j >= n) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int i = 0; i < m; ++i) {
        acc += a[i * k + p] * out_grad[i * n + j];
    }
    atomicAdd(&b_grad[p * n + j], acc);
}

__global__ void batch_matmul_grad_a_kernel(float* a_grad, const float* b, const float* out_grad,
                                           unsigned int batches, unsigned int heads, unsigned int m,
                                           unsigned int k, unsigned int n) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batches * heads * m * k;
    if (id >= total) {
        return;
    }
    unsigned int p = id % k;
    unsigned int i = (id / k) % m;
    unsigned int head = (id / ((long long)k * m)) % heads;
    unsigned int batch = id / ((long long)k * m * heads);
    float acc = 0.0f;
    for (unsigned int j = 0; j < n; ++j) {
        long long bi = ((long long)(batch * heads + head) * k + p) * n + j;
        long long oi = ((long long)(batch * heads + head) * m + i) * n + j;
        acc += out_grad[oi] * b[bi];
    }
    atomicAdd(&a_grad[id], acc);
}

__global__ void batch_matmul_grad_b_kernel(float* b_grad, const float* a, const float* out_grad,
                                           unsigned int batches, unsigned int heads, unsigned int m,
                                           unsigned int k, unsigned int n) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batches * heads * k * n;
    if (id >= total) {
        return;
    }
    unsigned int j = id % n;
    unsigned int p = (id / n) % k;
    unsigned int head = (id / ((long long)n * k)) % heads;
    unsigned int batch = id / ((long long)n * k * heads);
    float acc = 0.0f;
    for (unsigned int i = 0; i < m; ++i) {
        long long ai = ((long long)(batch * heads + head) * m + i) * k + p;
        long long oi = ((long long)(batch * heads + head) * m + i) * n + j;
        acc += a[ai] * out_grad[oi];
    }
    atomicAdd(&b_grad[id], acc);
}

__global__ void softmax_grad_kernel(float* a_grad, const float* out, const float* out_grad,
                                    unsigned int rows, unsigned int width) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned int base = row * width;
    float dot = 0.0f;
    for (unsigned int c = 0; c < width; ++c) {
        dot += out_grad[base + c] * out[base + c];
    }
    for (unsigned int c = 0; c < width; ++c) {
        atomicAdd(&a_grad[base + c], out[base + c] * (out_grad[base + c] - dot));
    }
}

__global__ void cross_entropy_grad_kernel(float* logits_grad, const float* logits,
                                          const float* targets, const float* out_grad,
                                          unsigned int rows, unsigned int vocab) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * vocab;
    if (id >= total) {
        return;
    }
    unsigned int row = id / vocab;
    unsigned int v = id % vocab;
    unsigned int base = row * vocab;
    float mx = -INFINITY;
    for (unsigned int c = 0; c < vocab; ++c) {
        mx = fmaxf(mx, logits[base + c]);
    }
    float denom = 0.0f;
    for (unsigned int c = 0; c < vocab; ++c) {
        denom += expf(logits[base + c] - mx);
    }
    float g = expf(logits[id] - mx) / denom;
    if (v == (unsigned int)targets[row]) {
        g -= 1.0f;
    }
    atomicAdd(&logits_grad[id], out_grad[0] * g / (float)rows);
}

__global__ void cross_entropy_grad_rows_kernel(float* logits_grad, const float* logits,
                                               const float* targets, const float* out_grad,
                                               unsigned int rows, unsigned int vocab) {
    unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned int tid = threadIdx.x;
    unsigned int base = row * vocab;
    __shared__ float scratch[256];

    float local_max = -INFINITY;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        local_max = fmaxf(local_max, logits[base + v]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float mx = scratch[0];

    float local_sum = 0.0f;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        local_sum += expf(logits[base + v] - mx);
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float denom = scratch[0];
    unsigned int target = (unsigned int)(targets[row]);
    float scale = out_grad[0] / (float)rows;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        float g = expf(logits[base + v] - mx) / denom;
        if (v == target) {
            g -= 1.0f;
        }
        logits_grad[base + v] += scale * g;
    }
}

__global__ void embedding_grad_kernel(float* weight_grad, const float* ids, const float* out_grad,
                                      unsigned int count, unsigned int dim) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)count * dim;
    if (id >= total) {
        return;
    }
    unsigned int token_index = id / dim;
    unsigned int d = id % dim;
    unsigned int token = (unsigned int)ids[token_index];
    atomicAdd(&weight_grad[token * dim + d], out_grad[id]);
}

__global__ void layernorm_grad_x_kernel(float* x_grad, const float* x, const float* scale,
                                        const float* out_grad, unsigned int rows, unsigned int width,
                                        float eps) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * width;
    if (id >= total) {
        return;
    }
    unsigned int row = id / width;
    unsigned int c = id % width;
    unsigned int base = row * width;
    float mean = 0.0f;
    for (unsigned int i = 0; i < width; ++i) {
        mean += x[base + i];
    }
    mean /= (float)width;
    float var = 0.0f;
    for (unsigned int i = 0; i < width; ++i) {
        float z = x[base + i] - mean;
        var += z * z;
    }
    float inv = rsqrtf(var / (float)width + eps);
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (unsigned int i = 0; i < width; ++i) {
        float xhat = (x[base + i] - mean) * inv;
        float dxhat = out_grad[base + i] * scale[i];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
    }
    float xhat = (x[id] - mean) * inv;
    float dxhat = out_grad[id] * scale[c];
    float g = ((float)width * dxhat - sum_dxhat - xhat * sum_dxhat_xhat) * inv / (float)width;
    atomicAdd(&x_grad[id], g);
}

__global__ void layernorm_grad_scale_shift_kernel(float* scale_grad, float* shift_grad,
                                                  const float* x, const float* out_grad,
                                                  unsigned int rows, unsigned int width, float eps) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)rows * width;
    if (id >= total) {
        return;
    }
    unsigned int row = id / width;
    unsigned int c = id % width;
    unsigned int base = row * width;
    float mean = 0.0f;
    for (unsigned int i = 0; i < width; ++i) {
        mean += x[base + i];
    }
    mean /= (float)width;
    float var = 0.0f;
    for (unsigned int i = 0; i < width; ++i) {
        float z = x[base + i] - mean;
        var += z * z;
    }
    float inv = rsqrtf(var / (float)width + eps);
    float xhat = (x[id] - mean) * inv;
    if (scale_grad != nullptr) {
        atomicAdd(&scale_grad[c], out_grad[id] * xhat);
    }
    if (shift_grad != nullptr) {
        atomicAdd(&shift_grad[c], out_grad[id]);
    }
}

__global__ void gelu_grad_kernel(float* x_grad, const float* x, const float* out_grad,
                                 long long count) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        float v = x[id];
        float u = 0.7978845608f * (v + 0.044715f * v * v * v);
        float th = tanhf(u);
        float du = 0.7978845608f * (1.0f + 3.0f * 0.044715f * v * v);
        float g = 0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * du;
        atomicAdd(&x_grad[id], out_grad[id] * g);
    }
}

__global__ void adamw_update_kernel(float* param, const float* grad, float* m, float* v,
                                    long long count, float lr, float weight_decay, float beta1,
                                    float beta2, float eps, float bias_correction1,
                                    float bias_correction2) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        float g = grad[id];
        m[id] = beta1 * m[id] + (1.0f - beta1) * g;
        v[id] = beta2 * v[id] + (1.0f - beta2) * g * g;
        float m_hat = m[id] / bias_correction1;
        float v_hat = v[id] / bias_correction2;
        param[id] -= lr * weight_decay * param[id];
        param[id] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

__global__ void causal_mask_kernel(const float* scores, float* out, unsigned int batches,
                                   unsigned int heads, unsigned int sequence_length, float mask_value) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batches * heads * sequence_length * sequence_length;
    if (id >= total) {
        return;
    }
    unsigned int j = id % sequence_length;
    unsigned int i = (id / sequence_length) % sequence_length;
    out[id] = (j > i) ? mask_value : scores[id];
}

__global__ void causal_mask_grad_kernel(float* scores_grad, const float* out_grad,
                                        unsigned int batches, unsigned int heads,
                                        unsigned int sequence_length) {
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batches * heads * sequence_length * sequence_length;
    if (id >= total) {
        return;
    }
    unsigned int j = id % sequence_length;
    unsigned int i = (id / sequence_length) % sequence_length;
    if (j <= i) {
        atomicAdd(&scores_grad[id], out_grad[id]);
    }
}

} // namespace llm::cuda::detail
