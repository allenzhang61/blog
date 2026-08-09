#pragma once

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_inference {

// 单个已上传到 CUDA device 的权重缓冲区。
class DeviceWeight {
public:
    // device 端权重数据指针。
    void * ptr = nullptr;
    // device 端权重缓冲区字节数。
    size_t bytes = 0;
    // cuBLAS 使用的数据类型。
    cudaDataType_t type = CUDA_R_32F;

    DeviceWeight() = default;
    // 释放 device 端权重数据。
    ~DeviceWeight();
    DeviceWeight(const DeviceWeight &) = delete;
    DeviceWeight & operator=(const DeviceWeight &) = delete;
    // 转移 device 指针所有权，避免重复 cudaFree。
    DeviceWeight(DeviceWeight && other) noexcept;
    // 释放当前缓冲区并接管 other 的 device 指针。
    DeviceWeight & operator=(DeviceWeight && other) noexcept;
};

} // namespace llm_inference
