//
// Created by zhangyoulun on 9/8/2026.
//

#include "gemm.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../common.h"
#include "kernel.cuh"
#include "utils/stats/ScopedTimer.h"

namespace {

cudaDataType_t cuda_type_of(DType dtype) {
    if (dtype == DType::BF16) return CUDA_R_16BF;
    if (dtype == DType::F16) return CUDA_R_16F;
    if (dtype == DType::F32) return CUDA_R_32F;
    throw std::runtime_error(std::string("gemm 不支持 dtype: ") + dtype_name(dtype));
}

// cublasGemmEx 要求激活与权重同 dtype；把权重 dtype 翻译成 kernel 约定的 lowp_type（0=bf16, 1=f16）。
int lowp_of(DType dtype) { return dtype == DType::F16 ? 1 : 0; }

} // namespace

// w * x = y
void gemm_weight(cublasHandle_t handle, const Tensor &weight,
                 const void *d_x, float *d_y,
                 int out_dim, int in_dim, size_t input_size, cudaDataType_t x_type,
                 const char *name) {
    const cudaDataType_t weight_type = cuda_type_of(weight.dtype_dequant);
    // cublasGemmEx 要求激活与权重同 dtype，否则 GEMM 结果错误。
    if (weight_type != x_type) {
        throw std::runtime_error("gemm_weight: weight.type 与 x_type 不一致");
    }
    // name 非空时埋点：以 weight.bytes 作为访存字节，供 Profiler 算有效带宽。
    // ScopedGpuTimer 在 Profiler 关闭或 name 为空时零开销。
    ScopedGpuTimer timer(name && name[0] ? name : std::string(), nullptr, weight.nbytes_dequant);
    const int token_count = static_cast<int>(input_size);

    // 纯 w*x=y，不累加
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 列主序视角：W 存为 [in_dim, out_dim]（lda=in_dim），OP_T 得 [out_dim, in_dim]；
    // X 为 [in_dim, tokens]（ldb=in_dim）；Y 为 [out_dim, tokens]（ldc=out_dim）。
    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,//A(weight)转置
            CUBLAS_OP_N,//B(d_in)不转置
            out_dim,
            token_count,
            in_dim,
            &alpha,
            weight.gpu_data_dequant,//权重
            weight_type,//权重的数据类型
            in_dim,
            d_x,//输入
            x_type,//输入的数据类型
            in_dim,
            &beta,
            d_y,//输出
            CUDA_R_32F,//返回值类型，即float
            out_dim,
            CUBLAS_COMPUTE_32F,//fp32累加（精度好）
            CUBLAS_GEMM_DEFAULT),//默认算法选择
        "cublasGemmEx 投影失败");
}

void float_to_lowp(const float *d_x, uint16_t *d_x_lowp, size_t n,
                    DType weight_dtype, void *stream) {
    launch_float_to_lowp(d_x, d_x_lowp, static_cast<int>(n), lowp_of(weight_dtype), stream);
}

GemmInput prepare_gemm_input(const float *d_x, uint16_t *d_x_lowp, size_t n,
                             DType weight_dtype, void *stream) {
    // F32 权重：cublasGemmEx 走 f32×f32，激活保持原始 float，无需压缩与额外拷贝。
    if (weight_dtype == DType::F32) {
        return GemmInput{d_x, CUDA_R_32F};
    }
    // F16/BF16 权重：把 float 激活压成权重 dtype 后再投影。
    float_to_lowp(d_x, d_x_lowp, n, weight_dtype, stream);
    return GemmInput{d_x_lowp, cuda_type_of(weight_dtype)};
}
