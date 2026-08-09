//
// Created by zhangyoulun on 9/8/2026.
//

#include "gemm.h"

#include "../common.h"
#include "../mem/CudaWeight.h"
#include "kernel.cuh"

// cublasGemmEx 要求激活与权重同 dtype；把权重 dtype 翻译成 kernel 约定的 lowp_type（0=bf16, 1=f16）。
static int lowp_of(cudaDataType_t type) { return type == CUDA_R_16F ? 1 : 0; }

void gemm_weight(cublasHandle_t handle, const CudaWeight &weight,
                 int out_dim, int in_dim,
                 const void *d_x, cudaDataType_t x_type, int tokens, float *d_y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 列主序视角：W 存为 [in_dim, out_dim]（lda=in_dim），OP_T 得 [out_dim, in_dim]；
    // X 为 [in_dim, tokens]（ldb=in_dim）；Y 为 [out_dim, tokens]（ldc=out_dim）。
    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_dim,
            tokens,
            in_dim,
            &alpha,
            weight.ptr,
            weight.type,
            in_dim,
            d_x,
            x_type,
            in_dim,
            &beta,
            d_y,
            CUDA_R_32F,
            out_dim,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx 投影失败");
}

void to_weight_lowp(const float *d_x, uint16_t *d_x_lowp, int n,
                    const CudaWeight &weight, void *stream) {
    launch_float_to_lowp(d_x, d_x_lowp, n, lowp_of(weight.type), stream);
}
