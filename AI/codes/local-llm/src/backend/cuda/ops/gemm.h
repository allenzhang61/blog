//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_GEMM_H
#define LOCAL_LLM_GEMM_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

class CudaWeight;

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
// 维度需显式传入：CudaWeight 只记录字节数与 dtype，不含 shape，
// 调用方从 WeightData.info->shape 取 out_dim / in_dim。
// x_type 指明激活数据类型（CUDA_R_32F 或 CUDA_R_16BF / CUDA_R_16F）；
// 权重类型取自 weight.type。

// 批量投影：Y[out_dim,tokens] = W[out_dim,in_dim] · X[in_dim,tokens]。
// tokens=1 即单 token 情形。
void gemm_weight(cublasHandle_t handle, const CudaWeight &weight,
                 int out_dim, int in_dim,
                 const void *d_x, cudaDataType_t x_type, int tokens, float *d_y);

#endif // LOCAL_LLM_GEMM_H
