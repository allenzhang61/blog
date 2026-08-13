//
// Created by zhangyoulun on 9/8/2026.
//

#include "gemm.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "../common.h"
#include "../mem/CudaWeight.h"
#include "kernel.cuh"
#include "utils/stats/ScopedTimer.h"

// cublasGemmEx 要求激活与权重同 dtype；把权重 dtype 翻译成 kernel 约定的 lowp_type（0=bf16, 1=f16）。
static int lowp_of(cudaDataType_t type) { return type == CUDA_R_16F ? 1 : 0; }

void gemm_weight(cublasHandle_t handle, const CudaWeight &weight,
                 int out_dim, int in_dim,
                 const void *d_x, cudaDataType_t x_type, size_t tokens, float *d_y,
                 const char *name) {
    // name 非空时埋点：以 weight.bytes 作为访存字节，供 Profiler 算有效带宽。
    // ScopedGpuTimer 在 Profiler 关闭或 name 为空时零开销。
    ScopedGpuTimer timer(name && name[0] ? name : std::string(), nullptr, weight.bytes);
    const int token_count = static_cast<int>(tokens);

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
            token_count,
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

void to_weight_lowp(const float *d_x, uint16_t *d_x_lowp, size_t n,
                    const CudaWeight &weight, void *stream) {
    launch_float_to_lowp(d_x, d_x_lowp, static_cast<int>(n), lowp_of(weight.type), stream);
}
