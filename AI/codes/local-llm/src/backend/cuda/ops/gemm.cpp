//
// Created by zhangyoulun on 9/8/2026.
//

#include "gemm.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "../common.h"
#include "utils/stats/ScopedTimer.h"

// w * x = y
void gemm_main(cublasHandle_t handle, const void *d_weight,
                 const void *d_x, float *d_y,
                 int out_dim, int in_dim, size_t input_size,
                 cudaDataType_t weight_type, cudaDataType_t x_type, size_t weight_bytes,
                 const char *name) {
    // cublasGemmEx 要求激活与权重同 dtype，否则 GEMM 结果错误。
    if (weight_type != x_type) {
        throw std::runtime_error("gemm_weight: g_weight.type 与 x_type 不一致");
    }
    // name 非空时埋点：以权重 bytes 作为访存字节，供 Profiler 算有效带宽。
    // ScopedGpuTimer 在 Profiler 关闭或 name 为空时零开销。name 为字面量指针，不拷贝。
    ScopedGpuTimer timer(name && name[0] ? name : nullptr, nullptr, weight_bytes);

    // 纯 w*x=y，不累加
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 绑定当前流：decode 阶段为非阻塞流，使 cuBLAS 调用可纳入 stream capture（CUDA Graph）。
    check_cublas(cublasSetStream(handle, get_current_cuda_stream()), "cublasSetStream 失败");

    // 列主序视角：W 存为 [in_dim, out_dim]（lda=in_dim），OP_T 得 [out_dim, in_dim]；
    // X 为 [in_dim, tokens]（ldb=in_dim）；Y 为 [out_dim, tokens]（ldc=out_dim）。
    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, //A(g_weight)转置
            CUBLAS_OP_N, //B(d_in)不转置
            out_dim,
            static_cast<int>(input_size),
            in_dim,
            &alpha,
            d_weight, //权重
            weight_type, //权重的数据类型
            in_dim,
            d_x, //输入
            x_type, //输入的数据类型
            in_dim,
            &beta,
            d_y, //输出
            CUDA_R_32F, //返回值类型，即float
            out_dim,
            CUBLAS_COMPUTE_32F, //fp32累加（精度好）
            CUBLAS_GEMM_DEFAULT), //默认算法选择
        "cublasGemmEx 投影失败");
}
