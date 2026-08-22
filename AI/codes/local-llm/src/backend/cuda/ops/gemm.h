//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_GEMM_H
#define LOCAL_LLM_GEMM_H

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS 矩阵乘封装：统一处理“权重 × 激活”这一类线性投影。
//
// 约定（与权重存储布局一致）：
//   - 权重 W 形状 [out_dim, in_dim]，行主序存储（safetensors 原样，每行一个输出通道）；
//   - 激活 X 视为 [in_dim, tokens]（每个 token 一列，in_dim 连续）；
//   - 输出 Y 形状 [out_dim, tokens]，float，每个 token 一列，out_dim 连续。
// cuBLAS 为列主序：把行主序 W[out_dim,in_dim] 视作列主序的 [in_dim,out_dim]，
// 取 CUBLAS_OP_T 还原成 [out_dim,in_dim] 再与 X[in_dim,tokens] 相乘，
// 得到 Y[out_dim,tokens]。计算精度固定 CUBLAS_COMPUTE_32F，输出恒为 CUDA_R_32F。
//
// 维度需显式传入：调用方从 tensor.shape 取 out_dim / in_dim。
// weight_type / x_type 分别指明权重与激活的 CUDA dtype；
// weight_bytes 由调用方按原始 tensor dtype 计算，用于 profiler 统计。

// 批量投影：Y[out_dim,tokens] = W[out_dim,in_dim] · X[in_dim,tokens]。
// tokens=1 即单 token 情形。
// name 非空时，用 ScopedGpuTimer 以该名埋点，并以权重字节数作为访存字节数
// （decode 为访存瓶颈，投影耗时主体即读取权重），供 Profiler 算有效带宽；
// 传空串（默认）则不埋点、零开销。
void gemm_main(cublasHandle_t handle, const void *d_weight,
                 const void *d_x, float *d_y,
                 int out_dim, int in_dim, size_t input_size,
                 cudaDataType_t weight_type, cudaDataType_t x_type, size_t weight_bytes,
                 const char *name = "");

#endif // LOCAL_LLM_GEMM_H
