#include "cuda_ops.h"
#include "cuda_common.h"

#include "../../core/cuda_kernels.h"

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_inference {

namespace {


void cuda_float_to_lowp_impl(const float * input, uint16_t * output, int n, cudaDataType_t type) {
    if (type == CUDA_R_16F) {
        launch_float_to_f16(input, output, n, nullptr);
    } else {
        launch_float_to_bf16(input, output, n, nullptr);
    }
}

void cuda_weight_matvec_to_device_impl(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y) {
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasGemmEx(
            cache.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_dim,
            1,
            in_dim,
            &alpha,
            device_weight.ptr,
            device_weight.type,
            in_dim,
            device_x,
            x_type,
            in_dim,
            &beta,
            device_y,
            CUDA_R_32F,
            out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx matvec 失败 " + weight.info->name);
}

void cuda_weight_batch_matvec_to_device_impl(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    int tokens,
    float * device_y) {
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasGemmEx(
            cache.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_dim,
            tokens,
            in_dim,
            &alpha,
            device_weight.ptr,
            device_weight.type,
            in_dim,
            device_x,
            x_type,
            in_dim,
            &beta,
            device_y,
            CUDA_R_32F,
            out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx batch matvec 失败 " + weight.info->name);
}

} // namespace

void cuda_float_to_lowp(const float * input, uint16_t * output, int n, cudaDataType_t type) {
    cuda_float_to_lowp_impl(input, output, n, type);
}

void cuda_weight_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y) {
    cuda_weight_matvec_to_device_impl(cache, weight, device_weight, device_x, x_type, device_y);
}

void cuda_weight_batch_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    int tokens,
    float * device_y) {
    cuda_weight_batch_matvec_to_device_impl(cache, weight, device_weight, device_x, x_type, tokens, device_y);
}

void * cuda_token_hidden_buffer(int slot, int hidden_size) {
    auto & cache = cuda_weight_cache();
    CudaScratchBuffer<float> & buffer = slot == 0 ? cache.token_hidden_a : cache.token_hidden_b;
    return buffer.ensure_bytes(static_cast<size_t>(hidden_size) * sizeof(float), slot == 0 ? "token hidden a" : "token hidden b");
}

void * cuda_token_id_buffer(int count) {
    auto & cache = cuda_weight_cache();
    cache.token_id_buffer.ensure_bytes(static_cast<size_t>(count) * sizeof(int), "token id buffer");
    return cache.token_id_buffer;
}

bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out) {
    if (!device_tokens || count < 0) {
        return false;
    }
    out.assign(static_cast<size_t>(count), 0);
    check_cuda(cudaMemcpy(out.data(), device_tokens, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy generated tokens 失败");
    return true;
}

bool cuda_synchronize_device() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize 失败");
    return true;
}

} // namespace llm_inference
