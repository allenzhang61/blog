#pragma once

#include "cache/CudaWeightCache.h"
#include "../../model/weights.h"

#include <cstdint>
#include <vector>

namespace llm_inference {

// 获取复用的 CUDA hidden buffer，slot 用于 current/next 双缓冲。
void * cuda_token_hidden_buffer(int slot, int hidden_size);

// 获取复用的 CUDA token id buffer；decode 存 generated ids，prefill 存 prompt ids。
void * cuda_token_id_buffer(int count);

// 将设备端生成 token ids 拷贝回主机 vector。
bool cuda_copy_generated_tokens_to_host(const void * device_tokens, int count, std::vector<int> & out);

// 同步当前 CUDA device，失败时返回 false。
bool cuda_synchronize_device();

// 将 float device buffer 转为 BF16/F16 device buffer。
void cuda_float_to_lowp(const float * input, uint16_t * output, int n, cudaDataType_t type);

// 使用 cuBLAS 执行单向量权重乘法，输出 float device buffer。
void cuda_weight_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    float * device_y);

// 使用 cuBLAS 执行 batch 权重乘法，输出 float device buffer。
void cuda_weight_batch_matvec_to_device(
    CudaWeightCache & cache,
    const WeightData & weight,
    DeviceWeight & device_weight,
    const void * device_x,
    cudaDataType_t x_type,
    int tokens,
    float * device_y);

} // namespace llm_inference
