#include "cuda_kernels.h"

#include <cuda_runtime.h>

namespace llm_inference {
namespace {

__device__ uint16_t float_to_bf16_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ float bf16_bits_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__global__ void silu_mul_kernel(const float * gate, const float * up, float * out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ void float_to_bf16_kernel(const float * input, uint16_t * output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = float_to_bf16_bits(input[i]);
}

__global__ void rms_norm_to_bf16_kernel(
    const float * input,
    const uint16_t * weight,
    uint16_t * output,
    int n,
    float eps,
    bool one_plus) {
    __shared__ float partial[256];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = input[i];
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(partial[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        const float w = bf16_bits_to_float(weight[i]);
        const float factor = one_plus ? (1.0f + w) : w;
        output[i] = float_to_bf16_bits(input[i] * scale * factor);
    }
}

} // namespace

void launch_silu_mul(const float * gate, const float * up, float * out, int n, void * stream) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, out, n);
}

void launch_float_to_bf16(const float * input, uint16_t * output, int n, void * stream) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    float_to_bf16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input, output, n);
}

void launch_rms_norm_to_bf16(
    const float * input,
    const uint16_t * weight,
    uint16_t * output,
    int n,
    float eps,
    bool one_plus,
    void * stream) {
    rms_norm_to_bf16_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(input, weight, output, n, eps, one_plus);
}

} // namespace llm_inference
